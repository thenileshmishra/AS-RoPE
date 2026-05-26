"""Run attention analysis on ALL checkpoints.

Usage:
    python -m pipeline.run_attention_batch
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from pipeline import paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch attention analysis")
    parser.add_argument("--langs", nargs="+", default=["de", "hi", "bn"])
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    ckpt_dir = paths.CHECKPOINT_DIR
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    tokenized_val = {
        "de": "processed_data_wmt14/tokenized/val_wmt14_en_de.pt",
        "hi": "processed_data/tokenized/val_hi_en.pt",
        "bn": "processed_data_bn/tokenized/val_bn_en.pt",
    }

    for subdir in sorted(ckpt_dir.iterdir()):
        if not subdir.is_dir():
            continue
        best_pt = subdir / "best.pt"
        if not best_pt.exists():
            continue

        run_name = subdir.name
        if run_name.startswith("_temp_") or run_name.startswith("_test"):
            continue

        # Only process paper checkpoints (skip legacy inconsistent ones)
        PAPER_NAMES = {
            'rope_de_s43', 'rope_de_s44', 'rope_hi_s42', 'rope_bn_s42',
            'adaptiverope_de_s43', 'adaptiverope_de_s44', 'adaptiverope_hi_s42', 'adaptiverope_bn_s42',
            'gatesonly_de_s42', 'gatesonly_de_s43', 'gatesonly_de_s44', 'gatesonly_hi_s42', 'gatesonly_bn_s42',
            'phasesonly_de_s42', 'phasesonly_de_s43', 'phasesonly_de_s44', 'phasesonly_hi_s42', 'phasesonly_bn_s42',
            'sinusoidal_de_correct', 'sinusoidal_de_s43', 'sinusoidal_de_s44', 'sinusoidal_bn_s42',
            'alibi_de_s42',
        }
        if run_name not in PAPER_NAMES:
            print(f"[attention] Skipping {run_name}: not a paper checkpoint")
            continue

        # Detect language
        lang = None
        for l in args.langs:
            if f"_{l}" in run_name or run_name.endswith(f"_{l}"):
                lang = l
                break
        if lang is None:
            continue

        val_path = tokenized_val.get(lang)
        if val_path is None:
            continue

        out_dir = paths.OUTPUTS_DIR / "analysis" / "attention" / run_name
        if (out_dir / "attention_analysis.json").exists():
            print(f"[attention] {run_name} already analyzed, skipping")
            continue

        print(f"\n[attention] Analyzing {run_name}...")
        result = subprocess.run([
            sys.executable, "-m", "src.attention_analysis",
            "--checkpoint", str(best_pt),
            "--tokenized-val", val_path,
            "--out-dir", str(out_dir),
            "--max-batches", str(args.max_batches),
            "--batch-size", str(args.batch_size),
            "--device", device,
        ])
        if result.returncode != 0:
            print(f"[attention] WARNING: {run_name} failed")

    print("\n[attention] All attention analyses complete.")


if __name__ == "__main__":
    main()
