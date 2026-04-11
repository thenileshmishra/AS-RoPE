"""Step 2 — Clean, split and cache the raw Samanantar dump.

Streams the raw TSV written by Step 1, applies cheap per-line filters
(length, length ratio, script, language, noise, copy) followed by a
sentence-embedding semantic similarity filter in chunks, deduplicates,
and stops once ``MAX_CLEAN_PAIRS`` high-quality pairs have been collected.
Deterministic train/val/test splits are then written back to Drive under
:mod:`pipeline.paths`.

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

# --- NEW FILTER --- lightweight semantic similarity config
SEMANTIC_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_MIN_SIM = 0.55
SEMANTIC_BATCH_SIZE = 256
STREAM_CHUNK_SIZE = 10_000

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


# --- NEW FILTER --- lightweight semantic similarity model loader
def _load_semantic_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[step2] sentence-transformers not installed; skipping semantic filter")
        return None
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(SEMANTIC_MODEL_NAME, device=device)
        print(f"[step2] loaded semantic model {SEMANTIC_MODEL_NAME} on {device}")
        return model
    except Exception as e:
        print(f"[step2] failed to load semantic model ({e}); skipping semantic filter")
        return None


def _semantic_mask(model, hi_list: list[str], en_list: list[str]) -> list[bool]:
    """Return per-pair keep/drop mask based on cosine similarity of embeddings."""
    emb_hi = model.encode(
        hi_list,
        batch_size=SEMANTIC_BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    emb_en = model.encode(
        en_list,
        batch_size=SEMANTIC_BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sims = (emb_hi * emb_en).sum(dim=1).cpu().tolist()
    return [s >= SEMANTIC_MIN_SIM for s in sims]


def clean_pairs_streaming(
    input_path: Path,
    max_clean: int = MAX_CLEAN_PAIRS,
    chunk_size: int = STREAM_CHUNK_SIZE,
) -> tuple[list[tuple[str, str]], dict]:
    """Stream the raw TSV, apply all filters, and return up to ``max_clean`` pairs.

    Processing pattern:
      1. Cheap per-line filters (length / script / noise / copy) on each row.
      2. Survivors are buffered up to ``chunk_size``, then scored by the
         semantic-similarity model in a single batched call.
      3. Duplicate checks run on canonicalised keys to catch casing/punct dupes.
      4. Early-exit once ``max_clean`` high-quality pairs have been kept.
    """
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
        "low_similarity": 0,
        "kept": 0,
    }
    seen_canonical: set[tuple[str, str]] = set()
    kept: list[tuple[str, str]] = []
    model = _load_semantic_model()

    def flush(buf: list[tuple[str, str]]) -> bool:
        """Semantic + dedup pass over a chunk. Returns True when max_clean hit."""
        if not buf:
            return False
        if model is not None:
            hi_list = [h for h, _ in buf]
            en_list = [e for _, e in buf]
            mask = _semantic_mask(model, hi_list, en_list)
        else:
            mask = [True] * len(buf)
        for (hi, en), keep in zip(buf, mask):
            if not keep:
                stats["low_similarity"] += 1
                continue
            canonical_pair = (_canonical(hi), _canonical(en))
            if canonical_pair in seen_canonical:
                stats["duplicate"] += 1
                continue
            seen_canonical.add(canonical_pair)
            kept.append((hi, en))
            stats["kept"] += 1
            if stats["kept"] >= max_clean:
                return True
        return False

    buffer: list[tuple[str, str]] = []
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
            buffer.append((hi, en))
            if len(buffer) >= chunk_size:
                reached = flush(buffer)
                buffer = []
                if stats["total"] % 100_000 == 0 or reached:
                    print(
                        f"[step2] streamed {stats['total']:,} lines | kept {stats['kept']:,}"
                    )
                if reached:
                    break
        else:
            if buffer:
                flush(buffer)

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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=STREAM_CHUNK_SIZE,
        help="Semantic-filter batch size for streaming (default: 10k)",
    )
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    if not paths.RAW_SAMANANTAR.exists():
        raise SystemExit(
            f"[step2] missing raw Samanantar at {paths.RAW_SAMANANTAR}. Run Step 1 first."
        )

    print(f"[step2] streaming raw pairs from {paths.RAW_SAMANANTAR}")
    cleaned, stats = clean_pairs_streaming(
        paths.RAW_SAMANANTAR, max_clean=args.max_clean, chunk_size=args.chunk_size
    )
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
        "semantic_model": SEMANTIC_MODEL_NAME,
        "semantic_min_sim": SEMANTIC_MIN_SIM,
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
