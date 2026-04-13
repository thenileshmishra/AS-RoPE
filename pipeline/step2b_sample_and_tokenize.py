"""Step 2b — Sample 1M pairs and pre-tokenize for the PE comparison study.

One-time preprocessing. Reads processed train/val TSVs, samples a deterministic
1M-pair training subset, tokenizes both splits with IndicBART in the Hi->En
direction using the same layout as MTDatasetV2 (``[<2hi> src] <sep> [<2en> tgt] <eos>``
with src tokens masked to -100), and saves the result as flat int32 tensors
with offsets. Training then avoids all tokenizer calls in the hot path.

Output files (under PROCESSED_DIR/tokenized/):
    train_1m_hi_en.pt
    val_hi_en.pt

Each file contains:
    {
        "input_ids_flat": int32 Tensor [total_tokens],
        "labels_flat":    int32 Tensor [total_tokens],
        "offsets":        int64 Tensor [n_examples + 1],
        "meta": {...},
    }
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm

from pipeline import paths
from src.dataset import load_pairs
from src.dataset_v2 import build_mt_example_v2
from src.tokenizer_utils import build_mt_tokenizer, verify_token_ids


LABEL_IGNORE = -100


def _tokenize_split(
    pairs: list[tuple[str, str]],
    tokenizer,
    max_seq_len: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    input_flat: list[int] = []
    label_flat: list[int] = []
    offsets: list[int] = [0]
    lengths: list[int] = []

    for src, tgt in tqdm(pairs, desc=desc, unit="pair"):
        ex = build_mt_example_v2(src, tgt, tokenizer, max_seq_len)
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        if not input_ids:
            continue
        input_flat.extend(input_ids)
        label_flat.extend(labels)
        offsets.append(len(input_flat))
        lengths.append(len(input_ids))

    input_t = torch.tensor(input_flat, dtype=torch.int32)
    label_t = torch.tensor(label_flat, dtype=torch.int32)
    offsets_t = torch.tensor(offsets, dtype=torch.int64)

    lengths_t = torch.tensor(lengths, dtype=torch.int64)
    stats = {
        "n_examples": len(lengths),
        "total_tokens": int(lengths_t.sum().item()),
        "seq_len_mean": float(lengths_t.float().mean().item()),
        "seq_len_p50": int(torch.quantile(lengths_t.float(), 0.50).item()),
        "seq_len_p95": int(torch.quantile(lengths_t.float(), 0.95).item()),
        "seq_len_p99": int(torch.quantile(lengths_t.float(), 0.99).item()),
        "seq_len_max": int(lengths_t.max().item()),
    }
    return input_t, label_t, offsets_t, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 2b: sample + pre-tokenize")
    parser.add_argument("--n-train", type=int, default=1_000_000)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--tokenizer", default="ai4bharat/IndicBART")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write the .pt files. Defaults to PROCESSED_DIR/tokenized.",
    )
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    if not paths.PROCESSED_TRAIN.exists() or not paths.PROCESSED_VAL.exists():
        raise SystemExit(
            f"[step2b] processed splits missing under {paths.PROCESSED_DIR}. Run Step 2 first."
        )

    output_dir = Path(args.output_dir) if args.output_dir else (paths.PROCESSED_DIR / "tokenized")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step2b] loading train pairs from {paths.PROCESSED_TRAIN}")
    t0 = time.monotonic()
    train_pairs_all = load_pairs(paths.PROCESSED_TRAIN)
    print(f"[step2b] loaded {len(train_pairs_all):,} train pairs in {time.monotonic() - t0:.1f}s")

    print(f"[step2b] loading val pairs from {paths.PROCESSED_VAL}")
    val_pairs = load_pairs(paths.PROCESSED_VAL)
    print(f"[step2b] loaded {len(val_pairs):,} val pairs")

    n_train = min(args.n_train, len(train_pairs_all))
    print(f"[step2b] sampling {n_train:,} train pairs deterministically (seed={args.seed})")
    rng = random.Random(args.seed)
    train_pairs = rng.sample(train_pairs_all, n_train)
    del train_pairs_all

    print(f"[step2b] sanity check — first train pair:")
    print(f"         src = {train_pairs[0][0][:120]}")
    print(f"         tgt = {train_pairs[0][1][:120]}")

    print(f"[step2b] building tokenizer: {args.tokenizer}")
    tokenizer = build_mt_tokenizer(args.tokenizer)
    verify_token_ids(tokenizer)

    meta_common = {
        "tokenizer": args.tokenizer,
        "direction": "hi_en",
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "pad_id": tokenizer.pad_token_id,
        "sep_id": tokenizer.sep_token_id,
        "eos_id": tokenizer.eos_token_id,
        "vocab_size": len(tokenizer),
    }

    print(f"[step2b] tokenizing {n_train:,} train pairs (max_seq_len={args.max_seq_len})")
    train_input, train_labels, train_offsets, train_stats = _tokenize_split(
        train_pairs, tokenizer, args.max_seq_len, desc="train"
    )
    train_out = output_dir / "train_1m_hi_en.pt"
    torch.save(
        {
            "input_ids_flat": train_input,
            "labels_flat": train_labels,
            "offsets": train_offsets,
            "meta": {**meta_common, "split": "train", **train_stats},
        },
        train_out,
    )
    print(f"[step2b] wrote {train_out} ({train_out.stat().st_size / 1e6:.1f} MB)")
    print(f"[step2b] train stats: {json.dumps(train_stats, indent=2)}")

    print(f"[step2b] tokenizing {len(val_pairs):,} val pairs")
    val_input, val_labels, val_offsets, val_stats = _tokenize_split(
        val_pairs, tokenizer, args.max_seq_len, desc="val"
    )
    val_out = output_dir / "val_hi_en.pt"
    torch.save(
        {
            "input_ids_flat": val_input,
            "labels_flat": val_labels,
            "offsets": val_offsets,
            "meta": {**meta_common, "split": "val", **val_stats},
        },
        val_out,
    )
    print(f"[step2b] wrote {val_out} ({val_out.stat().st_size / 1e6:.1f} MB)")
    print(f"[step2b] val stats: {json.dumps(val_stats, indent=2)}")

    summary = {
        "output_dir": str(output_dir),
        "train_file": str(train_out),
        "val_file": str(val_out),
        "train": train_stats,
        "val": val_stats,
        "meta": meta_common,
    }
    (output_dir / "tokenization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[step2b] done. Pick --max-seq-len ~= p99 = {train_stats['seq_len_p99']} for training.")


if __name__ == "__main__":
    main()
