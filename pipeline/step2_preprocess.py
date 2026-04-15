"""Step 2 — Basic clean, split and cache the raw Samanantar dump.

Streams the raw TSV written by Step 1, applies lightweight per-line filters
(length, length ratio, script, language, noise, copy), deduplicates, and
stops once ``MAX_CLEAN_PAIRS`` clean pairs have been collected.
Deterministic train/val/test splits are then written under :mod:`pipeline.paths`.

Usage:
    !python -m pipeline.step2_preprocess
    !python -m pipeline.step2_preprocess --max-clean 5000000

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
# --- NEW FILTER --- tightened length ratio 0.5/2.0 -> 0.7/1.5
MIN_LEN_RATIO = 0.7
MAX_LEN_RATIO = 1.5
# --- NEW FILTER --- tightened Devanagari script ratio 0.30 -> 0.6
MIN_DEVANAGARI_RATIO = 0.6
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05

# --- NEW FILTER --- hard cap on the number of clean pairs collected
MAX_CLEAN_PAIRS = 5_000_000

# --- NEW FILTER --- noise thresholds
MIN_LATIN_RATIO = 0.9
MAX_PUNCT_RATIO = 0.30
MAX_DIGIT_RATIO = 0.30
MAX_REPEAT_RUN = 4

# --- NEW FILTER --- copy/overlap detection threshold (Jaccard on tokens)
COPY_JACCARD_THRESHOLD = 0.5

_PUNCT_CHARS = set(".,;:!?\"'()[]{}<>|/\\-_=+*&^%$#@~`")


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def devanagari_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for ch in letters if "DEVANAGARI" in unicodedata.name(ch, ""))
    return dev / len(letters)


# --- NEW FILTER --- lightweight script-based language checks
def latin_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for ch in letters if "LATIN" in unicodedata.name(ch, ""))
    return latin / len(letters)


def is_hindi(text: str) -> bool:
    return devanagari_ratio(text) >= MIN_DEVANAGARI_RATIO


def is_english(text: str) -> bool:
    return latin_ratio(text) >= MIN_LATIN_RATIO


# --- NEW FILTER --- simple noise heuristics (URL/HTML/punct/digits/repeats)
def has_url_or_html(text: str) -> bool:
    lower = text.lower()
    if "http://" in lower or "https://" in lower or "www." in lower:
        return True
    return "<" in text and ">" in text


def has_excess_punct(text: str) -> bool:
    if not text:
        return False
    punct = sum(1 for ch in text if ch in _PUNCT_CHARS)
    return punct / len(text) > MAX_PUNCT_RATIO


def has_excess_digits(text: str) -> bool:
    if not text:
        return False
    digits = sum(1 for ch in text if ch.isdigit())
    return digits / len(text) > MAX_DIGIT_RATIO


def has_char_repeats(text: str, run_len: int = MAX_REPEAT_RUN) -> bool:
    run = 1
    prev = ""
    for ch in text:
        if ch == prev and not ch.isspace():
            run += 1
            if run >= run_len:
                return True
        else:
            run = 1
        prev = ch
    return False


def is_noisy(text: str) -> bool:
    return (
        has_url_or_html(text)
        or has_excess_punct(text)
        or has_excess_digits(text)
        or has_char_repeats(text)
    )


# --- NEW FILTER --- identical / high-overlap copy detection
def is_copy(hi: str, en: str) -> bool:
    if hi.lower() == en.lower():
        return True
    hi_tokens = set(hi.lower().split())
    en_tokens = set(en.lower().split())
    if not hi_tokens or not en_tokens:
        return False
    union = hi_tokens | en_tokens
    if not union:
        return False
    return len(hi_tokens & en_tokens) / len(union) > COPY_JACCARD_THRESHOLD


def _canonical(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return re.sub(r"[^\w\s]", "", text)


def _cheap_filter(hi: str, en: str, stats: dict) -> bool:
    """Apply all per-line (non-semantic) filters. Return True to keep."""
    hi_len = len(hi.split())
    en_len = len(en.split())
    if hi_len < MIN_TOKENS or en_len < MIN_TOKENS:
        stats["too_short"] += 1
        return False
    if hi_len > MAX_TOKENS or en_len > MAX_TOKENS:
        stats["too_long"] += 1
        return False
    ratio = hi_len / max(en_len, 1)
    if ratio < MIN_LEN_RATIO or ratio > MAX_LEN_RATIO:
        stats["length_ratio"] += 1
        return False
    # --- NEW FILTER --- language / script check on each side
    if not is_hindi(hi):
        stats["not_hindi"] += 1
        return False
    if not is_english(en):
        stats["not_english"] += 1
        return False
    # --- NEW FILTER --- noise heuristics
    if is_noisy(hi) or is_noisy(en):
        stats["noisy"] += 1
        return False
    # --- NEW FILTER --- copy / overlap
    if is_copy(hi, en):
        stats["copy"] += 1
        return False
    return True


def clean_pairs_streaming(
    input_path: Path,
    max_clean: int = MAX_CLEAN_PAIRS,
) -> tuple[list[tuple[str, str]], dict]:
    """Stream the raw TSV, apply basic filters, and return up to ``max_clean`` pairs."""
    stats = {
        "total": 0,
        "too_short": 0,
        "too_long": 0,
        "length_ratio": 0,
        "not_hindi": 0,
        "not_english": 0,
        "noisy": 0,
        "copy": 0,
        "duplicate": 0,
        "kept": 0,
    }
    seen_canonical: set[tuple[str, str]] = set()
    kept: list[tuple[str, str]] = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            stats["total"] += 1
            if "\t" not in line:
                continue
            hi, en = line.rstrip("\n").split("\t", maxsplit=1)
            hi = normalize_text(hi)
            en = normalize_text(en)
            if not hi or not en:
                continue
            if not _cheap_filter(hi, en, stats):
                continue
            canonical_pair = (_canonical(hi), _canonical(en))
            if canonical_pair in seen_canonical:
                stats["duplicate"] += 1
                continue
            seen_canonical.add(canonical_pair)
            kept.append((hi, en))
            stats["kept"] += 1

            if stats["total"] % 100_000 == 0:
                print(f"[step2] streamed {stats['total']:,} lines | kept {stats['kept']:,}")

            if stats["kept"] >= max_clean:
                break

    return kept[:max_clean], stats


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
    parser.add_argument(
        "--max-clean",
        type=int,
        default=MAX_CLEAN_PAIRS,
        help="Stop after collecting this many clean pairs (default: 5M)",
    )
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    if not paths.RAW_SAMANANTAR.exists():
        raise SystemExit(
            f"[step2] missing raw Samanantar at {paths.RAW_SAMANANTAR}. Run Step 1 first."
        )

    print(f"[step2] streaming raw pairs from {paths.RAW_SAMANANTAR}")
    cleaned, stats = clean_pairs_streaming(paths.RAW_SAMANANTAR, max_clean=args.max_clean)
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
        "min_latin_ratio": MIN_LATIN_RATIO,
        "max_clean_pairs": args.max_clean,
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
