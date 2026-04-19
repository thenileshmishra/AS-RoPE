"""Step 1 (WMT14) — Download English-German from HuggingFace datasets.

Downloads the WMT14 de-en dataset and saves flat TSV files (en<TAB>de per line):
  raw_data/wmt14/train.tsv    (~4.5 M sentence pairs)
  raw_data/wmt14/val.tsv      (newstest2013, ~3 K pairs)
  raw_data/wmt14/test.tsv     (newstest2014, ~3 K pairs — standard benchmark)

Requires:
    pip install datasets

Usage:
    python -m pipeline.step1_download_wmt14
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import paths


def _save_tsv(examples, out_path: Path, src_key: str = "en", tgt_key: str = "de") -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            pair = ex["translation"]
            src = pair[src_key].strip()
            tgt = pair[tgt_key].strip()
            if src and tgt:
                f.write(f"{src}\t{tgt}\n")
                n += 1
            if n % 500_000 == 0 and n > 0:
                print(f"  wrote {n:,} pairs -> {out_path}")
    return n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 1 WMT14: download En-De")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    args = parser.parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "[step1_wmt14] 'datasets' package not found. "
            "Install with: pip install datasets"
        )

    paths.WMT14_DIR.mkdir(parents=True, exist_ok=True)

    if paths.RAW_WMT14_TRAIN.exists():
        size = paths.RAW_WMT14_TRAIN.stat().st_size
        print(f"[step1_wmt14] train.tsv already exists ({size / 1e9:.2f} GB), skipping download.")
        print("[step1_wmt14] Delete raw_data/wmt14/ and re-run to force re-download.")
        return

    print("[step1_wmt14] downloading WMT14 de-en from HuggingFace (this may take 10-30 min)...")
    ds = load_dataset("wmt14", "de-en", cache_dir=args.cache_dir)

    print(f"[step1_wmt14] dataset splits: {list(ds.keys())}")
    print(f"[step1_wmt14] train size: {len(ds['train']):,}")

    print(f"[step1_wmt14] writing train -> {paths.RAW_WMT14_TRAIN}")
    n_train = _save_tsv(ds["train"], paths.RAW_WMT14_TRAIN)
    print(f"[step1_wmt14] train: {n_train:,} pairs ({paths.RAW_WMT14_TRAIN.stat().st_size / 1e9:.2f} GB)")

    val_split = "validation" if "validation" in ds else "valid"
    print(f"[step1_wmt14] writing val (newstest2013) -> {paths.RAW_WMT14_VAL}")
    n_val = _save_tsv(ds[val_split], paths.RAW_WMT14_VAL)
    print(f"[step1_wmt14] val: {n_val:,} pairs")

    print(f"[step1_wmt14] writing test (newstest2014) -> {paths.RAW_WMT14_TEST}")
    n_test = _save_tsv(ds["test"], paths.RAW_WMT14_TEST)
    print(f"[step1_wmt14] test: {n_test:,} pairs")

    print("[step1] done. Proceed to step2_tokenize.")


if __name__ == "__main__":
    main()
