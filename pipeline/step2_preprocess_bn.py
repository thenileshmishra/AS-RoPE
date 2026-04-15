"""Step 2 (Bengali) — Clean and split Samanantar Bn-En.

Same filters as step2_preprocess (length, ratio, noise, dedup) but with
Bengali script detection instead of Devanagari. Targets 500K clean pairs.

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
MAX_PUNCT_RATIO = 0.30
MAX_DIGIT_RATIO = 0.30
MAX_REPEAT_RUN = 4
COPY_JACCARD_THRESHOLD = 0.5
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


def _punct_ratio(text: str) -> float:
    punct = set(".,;:!?\"'()[]{}<>|/\\-_=+*&^%$#@~`")
    chars = [ch for ch in text if not ch.isspace()]
    return sum(1 for ch in chars if ch in punct) / len(chars) if chars else 0.0


def _digit_ratio(text: str) -> float:
    chars = [ch for ch in text if not ch.isspace()]
    return sum(1 for ch in chars if ch.isdigit()) / len(chars) if chars else 0.0


def _has_long_repeat(text: str, max_run: int) -> bool:
    prev, run = None, 0
    for ch in text:
        run = run + 1 if ch == prev else 1
        if run > max_run:
            return True
        prev = ch
    return False


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


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
    if _punct_ratio(bn) > MAX_PUNCT_RATIO or _punct_ratio(en) > MAX_PUNCT_RATIO:
        return False
    if _digit_ratio(bn) > MAX_DIGIT_RATIO or _digit_ratio(en) > MAX_DIGIT_RATIO:
        return False
    if _has_long_repeat(bn, MAX_REPEAT_RUN) or _has_long_repeat(en, MAX_REPEAT_RUN):
        return False
    if _jaccard(bn, en) > COPY_JACCARD_THRESHOLD:
        return False
    return True


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
            bn = _normalize(parts[0])
            en = _normalize(parts[1])
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
        "min_bengali_ratio": MIN_BENGALI_RATIO,
        "seed": SEED,
    }
    (paths.PROCESSED_DIR_BN / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[step2_bn] done. metadata -> {paths.PROCESSED_DIR_BN / 'metadata.json'}")


if __name__ == "__main__":
    main()
