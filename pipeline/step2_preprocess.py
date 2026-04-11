"""Step 2 — Clean, split and cache the raw Samanantar dump.

Reads the raw TSV written by Step 1 from Google Drive, applies cleaning
filters (length bounds, length ratio, Devanagari script check, dedupe),
produces deterministic train/val/test splits and writes them back to
Google Drive under :mod:`pipeline.paths`.

Usage:
    !python -m pipeline.step2_preprocess

This step assumes Step 1 has already written ``RAW_SAMANANTAR`` — if it is
missing the script exits with a clear error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from pipeline import paths


SEED = 42
MIN_TOKENS = 2
MAX_TOKENS = 128
MIN_LEN_RATIO = 0.5
MAX_LEN_RATIO = 2.0
MIN_DEVANAGARI_RATIO = 0.30
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def devanagari_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for ch in letters if "DEVANAGARI" in unicodedata.name(ch, ""))
    return dev / len(letters)


def read_raw_pairs(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            hi, en = line.rstrip("\n").split("\t", maxsplit=1)
            hi = normalize_text(hi)
            en = normalize_text(en)
            if hi and en:
                pairs.append((hi, en))
    return pairs


def clean_pairs(pairs: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], dict]:
    """Apply filters; return kept pairs and a rejection stats dict."""
    stats = {
        "total": len(pairs),
        "too_short": 0,
        "too_long": 0,
        "length_ratio": 0,
        "script_mismatch": 0,
        "duplicate": 0,
    }
    seen_exact: set[tuple[str, str]] = set()
    seen_canonical: set[tuple[str, str]] = set()
    kept: list[tuple[str, str]] = []

    def canonical(text: str) -> str:
        text = re.sub(r"\s+", " ", text.lower()).strip()
        return re.sub(r"[^\w\s]", "", text)

    for hi, en in pairs:
        hi_len = len(hi.split())
        en_len = len(en.split())
        if hi_len < MIN_TOKENS or en_len < MIN_TOKENS:
            stats["too_short"] += 1
            continue
        if hi_len > MAX_TOKENS or en_len > MAX_TOKENS:
            stats["too_long"] += 1
            continue
        ratio = hi_len / max(en_len, 1)
        if ratio < MIN_LEN_RATIO or ratio > MAX_LEN_RATIO:
            stats["length_ratio"] += 1
            continue
        if devanagari_ratio(hi) < MIN_DEVANAGARI_RATIO:
            stats["script_mismatch"] += 1
            continue
        canonical_pair = (canonical(hi), canonical(en))
        if (hi, en) in seen_exact or canonical_pair in seen_canonical:
            stats["duplicate"] += 1
            continue
        seen_exact.add((hi, en))
        seen_canonical.add(canonical_pair)
        kept.append((hi, en))

    stats["kept"] = len(kept)
    return kept, stats


def deterministic_split(
    pairs: list[tuple[str, str]], seed: int = SEED
) -> tuple[list, list, list]:
    """Hash-based split so the same row always lands in the same split."""
    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    val_threshold = int(TRAIN_RATIO * 1000)
    test_threshold = int((TRAIN_RATIO + VAL_RATIO) * 1000)
    for hi, en in pairs:
        digest = hashlib.md5(f"{seed}:{hi}".encode("utf-8")).hexdigest()
        bucket = int(digest[:3], 16) % 1000
        if bucket < val_threshold:
            train.append((hi, en))
        elif bucket < test_threshold:
            val.append((hi, en))
        else:
            test.append((hi, en))
    return train, val, test


def write_tsv(pairs: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for hi, en in pairs:
            f.write(f"{hi}\t{en}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 2: clean + split Samanantar")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    if not paths.RAW_SAMANANTAR.exists():
        raise SystemExit(
            f"[step2] missing raw Samanantar at {paths.RAW_SAMANANTAR}. Run Step 1 first."
        )

    print(f"[step2] reading raw pairs from {paths.RAW_SAMANANTAR}")
    raw = read_raw_pairs(paths.RAW_SAMANANTAR)
    print(f"[step2] loaded {len(raw):,} raw pairs")

    cleaned, stats = clean_pairs(raw)
    print(f"[step2] cleaning stats: {stats}")

    train, val, test = deterministic_split(cleaned, seed=args.seed)
    print(f"[step2] split sizes: train={len(train):,} val={len(val):,} test={len(test):,}")

    write_tsv(train, paths.PROCESSED_TRAIN)
    write_tsv(val, paths.PROCESSED_VAL)
    write_tsv(test, paths.PROCESSED_TEST)

    metadata = {
        "raw_path": str(paths.RAW_SAMANANTAR),
        "eval_path": str(paths.RAW_FLORES),
        "seed": args.seed,
        "min_tokens": MIN_TOKENS,
        "max_tokens": MAX_TOKENS,
        "min_len_ratio": MIN_LEN_RATIO,
        "max_len_ratio": MAX_LEN_RATIO,
        "min_devanagari_ratio": MIN_DEVANAGARI_RATIO,
        "stats": stats,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
    }
    paths.PROCESSED_METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[step2] metadata -> {paths.PROCESSED_METADATA}")
    print("[step2] done. Proceed to Step 3 (training).")


if __name__ == "__main__":
    main()
