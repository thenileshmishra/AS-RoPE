"""Step 2b (Bengali) — Pre-tokenize Bn-En pairs for the PE ablation.

Reads PROCESSED_TRAIN_BN / PROCESSED_VAL_BN, tokenizes with IndicBART using
the ``[<2bn> src] <sep> [<2en> tgt] <eos>`` layout, and writes flat int32
tensors to Drive. Identical structure to step2b but for Bengali.

Usage:
    !python -m pipeline.step2b_tokenize_bn \
        --n-train 450000 \
        --max-seq-len 192
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
from src.tokenizer_utils import build_mt_tokenizer, verify_token_ids


LABEL_IGNORE = -100
CHUNK = 50_000


def _tokenize_split(
    pairs: list[tuple[str, str]],
    tokenizer,
    max_seq_len: int,
    desc: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    sep_id = tokenizer.sep_token_id
    eos_id = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    # IndicBART has <2bn> for Bengali, <2en> for English
    bn_tag = "<2bn>" if "<2bn>" in vocab else ""
    en_tag = "<2en>" if "<2en>" in vocab else ""

    input_flat: list[int] = []
    label_flat: list[int] = []
    offsets: list[int] = [0]
    lengths: list[int] = []

    n = len(pairs)
    pbar = tqdm(total=n, desc=desc, unit="pair")
    for i in range(0, n, CHUNK):
        chunk = pairs[i: i + CHUNK]
        src_texts = [(f"{bn_tag} {s}".strip() if bn_tag else s) for s, _ in chunk]
        tgt_texts = [(f"{en_tag} {t}".strip() if en_tag else t) for _, t in chunk]
        src_enc = tokenizer(src_texts, add_special_tokens=False)["input_ids"]
        tgt_enc = tokenizer(tgt_texts, add_special_tokens=False)["input_ids"]

        budget = max(2, max_seq_len - 2)
        half = budget // 2
        for src_ids, tgt_ids in zip(src_enc, tgt_enc):
            if not src_ids or not tgt_ids:
                continue
            if len(src_ids) + len(tgt_ids) > budget:
                if len(src_ids) > half:
                    src_ids = src_ids[:half]
                remaining = budget - len(src_ids)
                if len(tgt_ids) > remaining:
                    tgt_ids = tgt_ids[:remaining]
            full = src_ids + [sep_id] + tgt_ids + [eos_id]
            input_ids = full[:-1]
            labels = full[1:]
            src_len = len(src_ids)
            labels = [LABEL_IGNORE] * src_len + labels[src_len:]
            input_flat.extend(input_ids)
            label_flat.extend(labels)
            offsets.append(len(input_flat))
            lengths.append(len(input_ids))
        pbar.update(len(chunk))
    pbar.close()

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
    return (
        torch.tensor(input_flat, dtype=torch.int32),
        torch.tensor(label_flat, dtype=torch.int32),
        torch.tensor(offsets, dtype=torch.int64),
        stats,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 2b-bn: tokenize Bengali-En pairs")
    parser.add_argument("--n-train", type=int, default=450_000)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--tokenizer", default="ai4bharat/IndicBART")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not paths.PROCESSED_TRAIN_BN.exists():
        raise SystemExit(
            f"[step2b_bn] Bengali train split missing: {paths.PROCESSED_TRAIN_BN}. "
            "Run step2_preprocess_bn first."
        )

    output_dir = paths.PROCESSED_DIR_BN / "tokenized"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step2b_bn] loading train pairs from {paths.PROCESSED_TRAIN_BN}")
    train_all = load_pairs(paths.PROCESSED_TRAIN_BN)
    val_pairs = load_pairs(paths.PROCESSED_VAL_BN)
    print(f"[step2b_bn] train={len(train_all):,}  val={len(val_pairs):,}")

    n_train = min(args.n_train, len(train_all))
    rng = random.Random(args.seed)
    train_pairs = rng.sample(train_all, n_train)
    del train_all

    print(f"[step2b_bn] building tokenizer: {args.tokenizer}")
    tokenizer = build_mt_tokenizer(args.tokenizer)
    verify_token_ids(tokenizer)

    meta_common = {
        "tokenizer": args.tokenizer,
        "direction": "bn_en",
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "pad_id": tokenizer.pad_token_id,
        "sep_id": tokenizer.sep_token_id,
        "eos_id": tokenizer.eos_token_id,
        "vocab_size": len(tokenizer),
    }

    print(f"[step2b_bn] tokenizing {n_train:,} train pairs ...")
    t0 = time.monotonic()
    tr_in, tr_lab, tr_off, tr_stats = _tokenize_split(
        train_pairs, tokenizer, args.max_seq_len, desc="bn-train"
    )
    train_out = output_dir / "train_bn_en.pt"
    torch.save({"input_ids_flat": tr_in, "labels_flat": tr_lab,
                "offsets": tr_off, "meta": {**meta_common, "split": "train", **tr_stats}},
               train_out)
    print(f"[step2b_bn] train -> {train_out}  ({train_out.stat().st_size/1e6:.1f} MB)  "
          f"{time.monotonic()-t0:.0f}s")
    print(f"[step2b_bn] train stats: {json.dumps(tr_stats, indent=2)}")

    print(f"[step2b_bn] tokenizing {len(val_pairs):,} val pairs ...")
    val_in, val_lab, val_off, val_stats = _tokenize_split(
        val_pairs, tokenizer, args.max_seq_len, desc="bn-val"
    )
    val_out = output_dir / "val_bn_en.pt"
    torch.save({"input_ids_flat": val_in, "labels_flat": val_lab,
                "offsets": val_off, "meta": {**meta_common, "split": "val", **val_stats}},
               val_out)
    print(f"[step2b_bn] val -> {val_out}  ({val_out.stat().st_size/1e6:.1f} MB)")

    summary = {"train_file": str(train_out), "val_file": str(val_out),
                "train": tr_stats, "val": val_stats, "meta": meta_common}
    (output_dir / "tokenization_summary_bn.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[step2b_bn] done.")


if __name__ == "__main__":
    main()
