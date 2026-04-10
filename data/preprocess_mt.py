"""Preprocessing and deterministic caching for Hindi-English MT data.

Reads raw parallel data (TSV or JSONL), cleans, deduplicates, splits
deterministically, and optionally produces cached tokenized tensors
for fast Day-3 training loading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from data.mt_dataset import deterministic_split, load_pairs


# ── cleaning ────────────────────────────────────────────────────────

def deduplicate_pairs(
    pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Remove exact (src, tgt) duplicates while preserving order."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


# ── I/O helpers ─────────────────────────────────────────────────────

def write_tsv(pairs: list[tuple[str, str]], path: Path) -> None:
    """Write pairs as a TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(f"{src}\t{tgt}\n")


def read_tsv_pairs(path: Path) -> list[tuple[str, str]]:
    """Read back a TSV written by write_tsv (already normalized)."""
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            src, tgt = line.split("\t", maxsplit=1)
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


# ── tokenized cache ─────────────────────────────────────────────────

def tokenize_and_cache(
    pairs: list[tuple[str, str]],
    tokenizer,
    max_length: int,
    out_path: Path,
) -> Path:
    """Tokenize all pairs and save as a .pt tensor cache.

    Saved dict keys: src_ids, tgt_ids (both LongTensor [N, max_length]).
    """
    src_ids_list: list[torch.Tensor] = []
    tgt_ids_list: list[torch.Tensor] = []

    for src, tgt in pairs:
        s = tokenizer(
            src,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        t = tokenizer(
            tgt,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        src_ids_list.append(s)
        tgt_ids_list.append(t)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if src_ids_list:
        src_ids = torch.stack(src_ids_list)
        tgt_ids = torch.stack(tgt_ids_list)
    else:
        src_ids = torch.zeros(0, max_length, dtype=torch.long)
        tgt_ids = torch.zeros(0, max_length, dtype=torch.long)
    torch.save({"src_ids": src_ids, "tgt_ids": tgt_ids}, out_path)
    return out_path


def build_tokenizer(name_or_path: str):
    """Load a HuggingFace tokenizer. Fails loudly if unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError(
            "transformers is required for tokenization. Install with: pip install transformers"
        )
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── main pipeline ───────────────────────────────────────────────────

def preprocess(
    raw_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    max_length: int = 128,
    tokenizer_name: str | None = None,
    dedupe: bool = True,
) -> dict:
    """Run the full preprocessing pipeline.

    Returns metadata dict (also saved to output_dir/metadata.json).
    """
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw input file not found: {raw_path}")

    # 1. Load and clean
    pairs = load_pairs(raw_path)
    loaded_count = len(pairs)
    if loaded_count == 0:
        raise ValueError(f"No valid pairs found in {raw_path}")

    if dedupe:
        pairs = deduplicate_pairs(pairs)
    cleaned_count = len(pairs)

    # 2. Deterministic split
    train, val, test = deterministic_split(pairs, seed=seed)

    # 3. Write split TSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": output_dir / "train.tsv",
        "val": output_dir / "val.tsv",
        "test": output_dir / "test.tsv",
    }
    write_tsv(train, split_paths["train"])
    write_tsv(val, split_paths["val"])
    write_tsv(test, split_paths["test"])

    # 4. Metadata
    metadata = {
        "raw_path": str(raw_path),
        "seed": seed,
        "dedupe": dedupe,
        "max_length": max_length,
        "tokenizer": tokenizer_name,
        "loaded_pairs": loaded_count,
        "cleaned_pairs": cleaned_count,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }

    # 5. Optional tokenized cache
    cache_paths: dict[str, str] = {}
    if tokenizer_name is not None:
        tokenizer = build_tokenizer(tokenizer_name)
        for split_name, split_pairs in [("train", train), ("val", val), ("test", test)]:
            cp = output_dir / f"{split_name}.pt"
            tokenize_and_cache(split_pairs, tokenizer, max_length, cp)
            cache_paths[split_name] = str(cp)

        # Save tokenizer config for reproducibility
        tok_meta = {
            "tokenizer_name": tokenizer_name,
            "max_length": max_length,
            "pad_token": tokenizer.pad_token,
            "vocab_size": tokenizer.vocab_size,
        }
        tok_meta_path = output_dir / "tokenizer_config.json"
        tok_meta_path.write_text(json.dumps(tok_meta, indent=2), encoding="utf-8")
        metadata["tokenizer_config_path"] = str(tok_meta_path)

    metadata["cache_paths"] = cache_paths

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


# ── CLI ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess Hindi-English parallel data for MT training."
    )
    parser.add_argument("raw_input", help="Path to raw TSV or JSONL file")
    parser.add_argument("output_dir", help="Directory for processed outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length (default: 128)")
    parser.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer name/path (optional)")
    parser.add_argument("--no-dedupe", action="store_true", help="Skip deduplication")

    args = parser.parse_args(argv)

    print(f"[preprocess] raw input : {args.raw_input}")
    print(f"[preprocess] output dir: {args.output_dir}")
    print(f"[preprocess] seed={args.seed}  max_length={args.max_length}  dedupe={not args.no_dedupe}")

    meta = preprocess(
        raw_path=args.raw_input,
        output_dir=args.output_dir,
        seed=args.seed,
        max_length=args.max_length,
        tokenizer_name=args.tokenizer,
        dedupe=not args.no_dedupe,
    )

    print(f"[preprocess] loaded  : {meta['loaded_pairs']}")
    print(f"[preprocess] cleaned : {meta['cleaned_pairs']}")
    print(f"[preprocess] train   : {meta['train_size']}")
    print(f"[preprocess] val     : {meta['val_size']}")
    print(f"[preprocess] test    : {meta['test_size']}")

    out = Path(args.output_dir)
    print(f"[preprocess] artifacts: {out / 'train.tsv'}, {out / 'val.tsv'}, {out / 'test.tsv'}")
    print(f"[preprocess] metadata : {out / 'metadata.json'}")
    if meta["cache_paths"]:
        for split, cp in meta["cache_paths"].items():
            print(f"[preprocess] cache    : {cp}")


if __name__ == "__main__":
    main()
