import os
import subprocess
import unicodedata

from datasets import load_dataset

# ========================
# Config
# ========================
RAW_OUT = "datasets/raw/samanantar_hi_en_raw.tsv"
PROCESSED_DIR = "datasets/processed/hien_v1"
# Use full dataset by default. Set env SAMANANTAR_SAMPLE_SIZE for quick local runs.
SAMPLE_SIZE = int(os.getenv("SAMANANTAR_SAMPLE_SIZE", "0"))
SEED = 42
MAX_LEN = 128
TOKENIZER = "gpt2"
MIN_TOKENS = 2
MAX_TOKENS = 256
MIN_LEN_RATIO = 0.5
MAX_LEN_RATIO = 2.0
MIN_DEVANAGARI_RATIO = 0.30
EVAL_TSV = "datasets/eval/flores200_hi_en_devtest.tsv"

# ========================
# Utils
# ========================
def norm(x):
    return x.strip()


def devanagari_ratio(text: str) -> float:
    """Fraction of letters that are Devanagari in the input text."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0

    def is_devanagari(ch: str) -> bool:
        name = unicodedata.name(ch, "")
        return "DEVANAGARI" in name

    dev = sum(1 for ch in letters if is_devanagari(ch))
    return dev / len(letters)


def canonicalize_for_overlap(text: str) -> str:
    text = " ".join(text.lower().split())
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace())


def train_eval_overlap_count(train_tsv: str, eval_tsv: str) -> tuple[int, int]:
    """Return overlap count and total train count based on canonicalized source."""
    train_src = set()
    with open(train_tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            src, _ = line.split("\t", maxsplit=1)
            if src:
                train_src.add(canonicalize_for_overlap(src))

    eval_src = set()
    with open(eval_tsv, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            src, _ = line.split("\t", maxsplit=1)
            if src:
                eval_src.add(canonicalize_for_overlap(src))

    overlap = len(train_src & eval_src)
    return overlap, len(train_src)

# ========================
# Step 1: Load dataset
# ========================
print("Loading Samanantar dataset...")
data = load_dataset("ai4bharat/samanantar", "hi", split="train")

# Sample to avoid huge load
if SAMPLE_SIZE > 0:
    sample_n = min(SAMPLE_SIZE, len(data))
    data = data.shuffle(seed=SEED).select(range(sample_n))
    print(f"Using sampled subset: {sample_n} rows")
else:
    print(f"Using full split: {len(data)} rows")

# ========================
# Step 2: Detect format
# ========================
src_col, tgt_col = None, None
probe_n = min(1000, len(data))

for ex in data.select(range(probe_n)):
    if "translation" in ex:
        tr = ex["translation"]
        if isinstance(tr, dict) and "hi" in tr and "en" in tr:
            src_col, tgt_col = "translation", "translation"
            break
    elif "src" in ex and "tgt" in ex:
        src_col, tgt_col = "src", "tgt"
        break
    elif "source" in ex and "target" in ex:
        src_col, tgt_col = "source", "target"
        break

if src_col is None:
    raise RuntimeError("Could not find translation columns")

print(f"Using columns: {src_col}, {tgt_col}")


def detect_hi_en_fields(dataset, src_field: str, tgt_field: str, n: int = 200) -> tuple[str, str]:
    """Detect which field is Hindi and which is English from sampled rows."""
    n = min(n, len(dataset))
    if n == 0:
        return src_field, tgt_field

    src_dev, tgt_dev = 0.0, 0.0
    sampled = dataset.select(range(n))
    for ex in sampled:
        if src_field == "translation":
            tr = ex["translation"]
            src_text = norm(tr.get("hi", ""))
            tgt_text = norm(tr.get("en", ""))
        else:
            src_text = norm(ex.get(src_field, ""))
            tgt_text = norm(ex.get(tgt_field, ""))
        src_dev += devanagari_ratio(src_text)
        tgt_dev += devanagari_ratio(tgt_text)

    # Higher Devanagari side is Hindi.
    if src_dev >= tgt_dev:
        return src_field, tgt_field
    return tgt_field, src_field


hi_field, en_field = detect_hi_en_fields(data, src_col, tgt_col)
print(f"Detected language mapping: hi={hi_field}, en={en_field}")

# ========================
# Step 3: Clean + Save raw TSV
# ========================
os.makedirs(os.path.dirname(RAW_OUT), exist_ok=True)

seen = set()
kept = 0
reject_stats = {
    "empty": 0,
    "too_short": 0,
    "too_long": 0,
    "length_ratio": 0,
    "script_mismatch": 0,
    "duplicate": 0,
    "parse_error": 0,
}

print("Cleaning and writing raw TSV...")

with open(RAW_OUT, "w", encoding="utf-8") as f:
    for ex in data:
        try:
            if src_col == "translation":
                tr = ex["translation"]
                src = norm(tr.get("hi", ""))
                tgt = norm(tr.get("en", ""))
            else:
                src = norm(ex.get(hi_field, ""))
                tgt = norm(ex.get(en_field, ""))

            # Cleaning rules
            if not src or not tgt:
                reject_stats["empty"] += 1
                continue
            src_len = len(src.split())
            tgt_len = len(tgt.split())

            if src_len < MIN_TOKENS or tgt_len < MIN_TOKENS:
                reject_stats["too_short"] += 1
                continue
            if src_len > MAX_TOKENS or tgt_len > MAX_TOKENS:
                reject_stats["too_long"] += 1
                continue

            len_ratio = src_len / max(tgt_len, 1)
            if len_ratio < MIN_LEN_RATIO or len_ratio > MAX_LEN_RATIO:
                reject_stats["length_ratio"] += 1
                continue

            if devanagari_ratio(src) < MIN_DEVANAGARI_RATIO:
                reject_stats["script_mismatch"] += 1
                continue

            key = (src, tgt)
            if key in seen:
                reject_stats["duplicate"] += 1
                continue
            seen.add(key)

            f.write(f"{src}\t{tgt}\n")
            kept += 1

        except Exception:
            reject_stats["parse_error"] += 1
            continue

print(f"Wrote {kept} cleaned pairs -> {RAW_OUT}")
for k, v in reject_stats.items():
    print(f"Rejected {k}: {v}")

# ========================
# Step 4: Run preprocessing
# ========================
print("Running preprocessing...")

os.makedirs(PROCESSED_DIR, exist_ok=True)

cmd = [
    "python", "-m", "data.preprocess_mt",
    RAW_OUT,
    PROCESSED_DIR,
    "--seed", str(SEED),
    "--max-length", str(MAX_LEN),
    "--tokenizer", TOKENIZER
]

subprocess.run(cmd, check=True)

train_tsv = os.path.join(PROCESSED_DIR, "train.tsv")
if os.path.exists(train_tsv) and os.path.exists(EVAL_TSV):
    overlap, train_total = train_eval_overlap_count(train_tsv, EVAL_TSV)
    pct = (100.0 * overlap / train_total) if train_total else 0.0
    print(
        f"Train/eval source overlap: {overlap}/{train_total} "
        f"({pct:.2f}%) against {EVAL_TSV}"
    )
    if overlap > 0:
        print("WARNING: Non-zero train/eval source overlap detected.")
else:
    print("Overlap check skipped (train/eval TSV missing).")

print("All steps completed successfully.")