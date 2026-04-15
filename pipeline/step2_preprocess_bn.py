"""Step 2 (Bengali) — Quick clean and split Samanantar Bn-En.

Applies lightweight per-line checks only (length, ratio, script, dedup)
to keep preprocessing fast and simple.

Usage:
    !python -m pipeline.step2_preprocess_bn
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from pathlib import Path

from pipeline import paths


SEED = 42
MIN_TOKENS = 2
MAX_TOKENS = 128
MIN_LEN_RATIO = 0.7
MAX_LEN_RATIO = 1.5
MIN_BENGALI_RATIO = 0.6     # src must be ≥60% Bengali script
MIN_LATIN_RATIO = 0.9       # tgt (English) must be ≥90% Latin
MAX_CLEAN_PAIRS = 500_000   # 500K — enough for a PE ablation paper
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _bengali_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    bn = sum(1 for ch in letters if "BENGALI" in unicodedata.name(ch, ""))
    return bn / len(letters)


def _latin_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if "LATIN" in unicodedata.name(ch, "")) / len(letters)


def _is_clean(bn: str, en: str) -> bool:
    bn_tok = bn.split()
    en_tok = en.split()
    if not (MIN_TOKENS <= len(bn_tok) <= MAX_TOKENS):
        return False
    if not (MIN_TOKENS <= len(en_tok) <= MAX_TOKENS):
        return False
    ratio = len(bn_tok) / max(len(en_tok), 1)
    if not (MIN_LEN_RATIO <= ratio <= MAX_LEN_RATIO):
        return False
    if _bengali_ratio(bn) < MIN_BENGALI_RATIO:
        return False
    if _latin_ratio(en) < MIN_LATIN_RATIO:
        return False
    return True


def _orient_bn_en(left: str, right: str) -> tuple[str, str]:
    """Return (bn, en), auto-swapping when raw columns are EN->BN."""
    left_bn = _bengali_ratio(left)
    right_bn = _bengali_ratio(right)
    left_lat = _latin_ratio(left)
    right_lat = _latin_ratio(right)

    # Strong preference for the side that looks Bengali vs Latin.
    if left_bn >= right_bn and right_lat >= left_lat:
        return left, right
    if right_bn > left_bn and left_lat > right_lat:
        return right, left

    # Fallback: keep original order when ambiguous.
    return left, right


def main() -> None:
    raw_tsv = paths.RAW_SAMANANTAR_BN
    if not raw_tsv.exists():
        raise SystemExit(f"[step2_bn] raw data missing: {raw_tsv}. Run step1_download_bn first.")

    paths.PROCESSED_DIR_BN.mkdir(parents=True, exist_ok=True)

    print(f"[step2_bn] reading {raw_tsv} ...")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    skipped = 0

    with open(raw_tsv, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            left = _normalize(parts[0])
            right = _normalize(parts[1])
            bn, en = _orient_bn_en(left, right)
            if not _is_clean(bn, en):
                skipped += 1
                continue
            key = hashlib.md5(f"{bn}|{en}".encode()).hexdigest()
            if key in seen:
                skipped += 1
                continue
            seen.add(key)
            pairs.append((bn, en))
            if len(pairs) >= MAX_CLEAN_PAIRS:
                break

    print(f"[step2_bn] clean pairs: {len(pairs):,} | skipped: {skipped:,}")

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    def _write(path: Path, subset: list[tuple[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for bn, en in subset:
                f.write(f"{bn}\t{en}\n")
        print(f"[step2_bn] wrote {len(subset):,} -> {path}")

    _write(paths.PROCESSED_TRAIN_BN, pairs[:n_train])
    _write(paths.PROCESSED_VAL_BN, pairs[n_train: n_train + n_val])

    meta = {
        "total_clean": n,
        "n_train": n_train,
        "n_val": n_val,
        "max_clean_pairs": MAX_CLEAN_PAIRS,
        "min_len_ratio": MIN_LEN_RATIO,
        "max_len_ratio": MAX_LEN_RATIO,
        "min_bengali_ratio": MIN_BENGALI_RATIO,
        "min_latin_ratio": MIN_LATIN_RATIO,
        "seed": SEED,
    }
    (paths.PROCESSED_DIR_BN / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[step2_bn] done. metadata -> {paths.PROCESSED_DIR_BN / 'metadata.json'}")


if __name__ == "__main__":
    main()
