"""Step 2 — Pre-tokenize WMT14 En-De for encoder-decoder training.

Produces per-split .pt files with separate src and tgt flat buffers:
    src_flat (int32), src_offsets (int64 n+1)
    tgt_flat (int32), tgt_offsets (int64 n+1)
    meta: pad_id, bos_id, eos_id, vocab_size, seq-len stats, etc.

Output dir: processed_data_wmt14/tokenized/
    train_wmt14_en_de.pt
    val_wmt14_en_de.pt
    test_wmt14_en_de.pt (val of WMT14 = newstest2013; test = newstest2014)

Usage:
    python -m pipeline.step2_tokenize
    python -m pipeline.step2_tokenize --max-train 500000   # sanity subset
    python -m pipeline.step2_tokenize --max-seq-len 128
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from pipeline import paths
from src.tokenizer_utils import build_mt_tokenizer, verify_token_ids


TOKENIZER_NAME = "Helsinki-NLP/opus-mt-en-de"
CHUNK = 50_000


def _load_pairs(tsv_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def _tokenize_split(
    pairs: list[tuple[str, str]],
    tokenizer,
    max_seq_len: int,
    eos_id: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Tokenize pairs independently. Src gets EOS appended; tgt does NOT
    (collator will add BOS to tgt_in and EOS to tgt_out)."""
    src_flat: list[int] = []
    tgt_flat: list[int] = []
    src_off: list[int] = [0]
    tgt_off: list[int] = [0]
    src_lens: list[int] = []
    tgt_lens: list[int] = []
    skipped = 0

    n = len(pairs)
    pbar = tqdm(total=n, desc=desc, unit="pair")
    src_budget = max(1, max_seq_len - 1)  # leave 1 for EOS
    tgt_budget = max(1, max_seq_len - 1)  # collator will add BOS + EOS, we pre-truncate

    for i in range(0, n, CHUNK):
        chunk = pairs[i : i + CHUNK]
        src_texts = [s for s, _ in chunk]
        tgt_texts = [t for _, t in chunk]
        src_enc = tokenizer(src_texts, add_special_tokens=False)["input_ids"]
        tgt_enc = tokenizer(tgt_texts, add_special_tokens=False)["input_ids"]

        for src_ids, tgt_ids in zip(src_enc, tgt_enc):
            if not src_ids or not tgt_ids:
                skipped += 1
                continue
            if len(src_ids) > src_budget:
                src_ids = src_ids[:src_budget]
            if len(tgt_ids) > tgt_budget - 1:  # -1 extra headroom for EOS added later
                tgt_ids = tgt_ids[: tgt_budget - 1]

            src_ids = src_ids + [eos_id]  # encoder input ends with EOS

            src_flat.extend(src_ids)
            tgt_flat.extend(tgt_ids)
            src_off.append(len(src_flat))
            tgt_off.append(len(tgt_flat))
            src_lens.append(len(src_ids))
            tgt_lens.append(len(tgt_ids))
        pbar.update(len(chunk))
    pbar.close()

    if skipped:
        print(f"[tokenize:{desc}] skipped {skipped} empty pairs")

    src_t = torch.tensor(src_flat, dtype=torch.int32)
    tgt_t = torch.tensor(tgt_flat, dtype=torch.int32)
    src_off_t = torch.tensor(src_off, dtype=torch.int64)
    tgt_off_t = torch.tensor(tgt_off, dtype=torch.int64)

    sl = torch.tensor(src_lens, dtype=torch.int64).float()
    tl = torch.tensor(tgt_lens, dtype=torch.int64).float()
    stats = {
        "n_examples": len(src_lens),
        "src_len_mean": float(sl.mean().item()),
        "src_len_p95": float(torch.quantile(sl, 0.95).item()),
        "src_len_p99": float(torch.quantile(sl, 0.99).item()),
        "src_len_max": float(sl.max().item()),
        "tgt_len_mean": float(tl.mean().item()),
        "tgt_len_p95": float(torch.quantile(tl, 0.95).item()),
        "tgt_len_p99": float(torch.quantile(tl, 0.99).item()),
        "tgt_len_max": float(tl.max().item()),
    }
    return src_t, src_off_t, tgt_t, tgt_off_t, stats


def _bos_id_for(tokenizer) -> int:
    """Marian/opus-mt: decoder_start_token = pad_token. Use pad_id as BOS."""
    bid = getattr(tokenizer, "bos_token_id", None)
    return int(bid) if bid is not None else int(tokenizer.pad_token_id)


def _save_split(
    split_name: str,
    pairs: list[tuple[str, str]],
    tokenizer,
    max_seq_len: int,
    eos_id: int,
    bos_id: int,
    output_dir: Path,
) -> tuple[Path, dict]:
    src_t, src_off_t, tgt_t, tgt_off_t, stats = _tokenize_split(
        pairs, tokenizer, max_seq_len, eos_id, desc=split_name
    )
    meta = {
        "tokenizer": TOKENIZER_NAME,
        "direction": "en_de",
        "tgt_lang": "de",
        "max_seq_len": max_seq_len,
        "pad_id": int(tokenizer.pad_token_id),
        "eos_id": int(eos_id),
        "bos_id": int(bos_id),
        "vocab_size": len(tokenizer),
        "split": split_name,
        **stats,
    }
    out = output_dir / f"{split_name}_wmt14_en_de.pt"
    torch.save({
        "src_flat": src_t,
        "src_offsets": src_off_t,
        "tgt_flat": tgt_t,
        "tgt_offsets": tgt_off_t,
        "meta": meta,
    }, out)
    size = out.stat().st_size
    unit = "GB" if size > 1e9 else "MB"
    scale = 1e9 if size > 1e9 else 1e6
    print(f"[step2] wrote {out} ({size / scale:.2f} {unit})")
    return out, meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tokenize WMT14 En-De as pairs")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--tokenizer", default=TOKENIZER_NAME)
    parser.add_argument("--max-train", type=int, default=None,
                        help="Cap training pairs (default: all ~4.5M)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    for p in (paths.RAW_WMT14_TRAIN, paths.RAW_WMT14_VAL, paths.RAW_WMT14_TEST):
        if not p.exists():
            raise SystemExit(f"[step2] missing {p}. Run step1_download_wmt14 first.")

    output_dir = Path(args.output_dir) if args.output_dir else paths.TOKENIZED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step2] building tokenizer: {args.tokenizer}")
    tokenizer = build_mt_tokenizer(args.tokenizer)
    verify_token_ids(tokenizer)
    eos_id = int(tokenizer.eos_token_id)
    bos_id = _bos_id_for(tokenizer)

    print(f"[step2] loading train pairs from {paths.RAW_WMT14_TRAIN}")
    t0 = time.monotonic()
    train_pairs = _load_pairs(paths.RAW_WMT14_TRAIN)
    print(f"[step2] loaded {len(train_pairs):,} train pairs in {time.monotonic() - t0:.1f}s")
    if args.max_train and args.max_train < len(train_pairs):
        train_pairs = train_pairs[: args.max_train]
        print(f"[step2] capped train to {len(train_pairs):,} (--max-train)")

    val_pairs = _load_pairs(paths.RAW_WMT14_VAL)
    test_pairs = _load_pairs(paths.RAW_WMT14_TEST)
    print(f"[step2] val={len(val_pairs):,}  test={len(test_pairs):,}")

    results = {}
    for name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        print(f"[step2] tokenizing {name} ({len(pairs):,})")
        out_path, meta = _save_split(name, pairs, tokenizer, args.max_seq_len,
                                      eos_id, bos_id, output_dir)
        results[name] = {"path": str(out_path), **meta}

    (output_dir / "tokenization_summary.json").write_text(json.dumps(results, indent=2))
    print(f"[step2] wrote {output_dir / 'tokenization_summary.json'}")
    p99s = results["train"]["src_len_p99"], results["train"]["tgt_len_p99"]
    print(f"[step2] train src p99={p99s[0]:.0f}  tgt p99={p99s[1]:.0f}  "
          f"(using max_seq_len={args.max_seq_len})")


if __name__ == "__main__":
    main()
