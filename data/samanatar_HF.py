import os
import subprocess
from datasets import load_dataset

# ========================
# Config
# ========================
RAW_OUT = "datasets/raw/samanantar_hi_en_raw.tsv"
PROCESSED_DIR = "datasets/processed/hien_v1"
SAMPLE_SIZE = 5000   # change if needed
SEED = 42
MAX_LEN = 128
TOKENIZER = "gpt2"

# ========================
# Utils
# ========================
def norm(x):
    return x.strip()

# ========================
# Step 1: Load dataset
# ========================
print("Loading Samanantar dataset...")
data = load_dataset("ai4bharat/samanantar", "hi", split="train")

# Sample to avoid huge load
data = data.shuffle(seed=SEED).select(range(SAMPLE_SIZE))

# ========================
# Step 2: Detect format
# ========================
src_col, tgt_col = None, None

for ex in data.select(range(1000)):
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

# ========================
# Step 3: Clean + Save raw TSV
# ========================
os.makedirs(os.path.dirname(RAW_OUT), exist_ok=True)

seen = set()
kept = 0

print("Cleaning and writing raw TSV...")

with open(RAW_OUT, "w", encoding="utf-8") as f:
    for ex in data:
        try:
            if src_col == "translation":
                tr = ex["translation"]
                src = norm(tr.get("hi", ""))
                tgt = norm(tr.get("en", ""))
            else:
                src = norm(ex.get(src_col, ""))
                tgt = norm(ex.get(tgt_col, ""))

            # Cleaning rules
            if not src or not tgt:
                continue
            if len(src.split()) < 2 or len(tgt.split()) < 2:
                continue
            if len(src.split()) > 256 or len(tgt.split()) > 256:
                continue

            key = (src, tgt)
            if key in seen:
                continue
            seen.add(key)

            f.write(f"{src}\t{tgt}\n")
            kept += 1

        except Exception:
            continue

print(f"Wrote {kept} cleaned pairs -> {RAW_OUT}")

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

print("All steps completed successfully.")