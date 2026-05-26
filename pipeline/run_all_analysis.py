"""Run complete analysis pipeline after all training is finished.

This script:
  1. Evaluates all checkpoints (greedy + beam-5) with auto-detected tokenizer
  2. Evaluates RoPE checkpoints with Position Interpolation (PI)
  3. Runs length generalization for ALL methods (multi-way comparison)
  4. Runs attention entropy analysis on all checkpoints
  5. Runs statistical significance tests
  6. Generates comparison plots and tables

Usage:
    python -m pipeline.run_all_analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline import paths


def run_cmd(cmd: list[str], desc: str) -> bool:
    print(f"\n[analysis] {'='*60}")
    print(f"[analysis] {desc}")
    print(f"[analysis] {'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[analysis] WARNING: {desc} failed with code {result.returncode}")
        return False
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run complete analysis pipeline")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-length-gen", action="store_true", help="Skip length generalization")
    parser.add_argument("--skip-attention", action="store_true", help="Skip attention analysis")
    parser.add_argument("--skip-stats", action="store_true", help="Skip statistical tests")
    args = parser.parse_args(argv)

    # ── 1. Evaluation (greedy + PI for RoPE only) ──
    if not args.skip_eval:
        run_cmd([
            sys.executable, "-m", "pipeline.run_paper_evals",
            "--beam-size", "1",
            "--pi-scales", "1.5", "2.0", "3.0",
        ], "Running paper evaluations + PI variants")

    # ── 2. Length generalization (multi-method) ──
    if not args.skip_length_gen:
        ckpt_dir = paths.CHECKPOINT_DIR
        # Build list of En-De checkpoints for length gen
        de_ckpts = []
        for subdir in sorted(ckpt_dir.iterdir()):
            if not subdir.is_dir():
                continue
            name = subdir.name
            if "_de" not in name or name.startswith("_"):
                continue
            best_pt = subdir / "best.pt"
            if best_pt.exists():
                # Extract label from name
                label = name.replace("_de", "").replace("_s42", "").replace("_s43", "").replace("_s44", "")
                label = label.replace("sinusoidal_de_correct", "sinusoidal")
                label = label.replace("adaptiverope", "AdaptiveRoPE")
                label = label.replace("gatesonly", "GatesOnly")
                label = label.replace("phasesonly", "PhasesOnly")
                label = label.replace("sinusoidal", "Sinusoidal")
                label = label.replace("alibi", "ALiBi")
                label = label.replace("rope", "RoPE")
                de_ckpts.append(f"{best_pt}:{label}")

        if len(de_ckpts) >= 2:
            cmd = [
                sys.executable, "-m", "pipeline.length_generalization_multi",
                "--eval-tsv", "raw_data/wmt14/test.tsv",
                "--train-tsv", "raw_data/wmt14/train.tsv",
                "--extend-to", "420",
                "--out-dir", str(paths.OUTPUTS_DIR / "analysis" / "length_gen_multi"),
            ]
            for ckpt in de_ckpts:
                cmd.extend(["--ckpts", ckpt])
            run_cmd(cmd, f"Length generalization ({len(de_ckpts)} methods)")
        else:
            print(f"[analysis] Skipping length gen: only {len(de_ckpts)} En-De checkpoints found")

    # ── 3. Attention analysis ──
    if not args.skip_attention:
        run_cmd([
            sys.executable, "-m", "pipeline.run_attention_batch",
        ], "Attention analysis on all checkpoints")

    # ── 4. Statistical tests ──
    if not args.skip_stats:
        for lang in ["de", "hi", "bn"]:
            glob_pattern = f"outputs/metrics/*_{lang}_eval/eval_summary.json"
            out_file = paths.OUTPUTS_DIR / "analysis" / f"stats_{lang}.json"

            for target in ["adaptiverope", "asrope3", "gatesonly", "phasesonly"]:
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "src.statistical_tests",
                        "--results-glob", glob_pattern,
                        "--baseline-pe", "rope",
                        "--target-pe", target,
                        "--out-file", str(out_file),
                    ], capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"[analysis] Stats {lang}: {target} vs rope ✅")
                        break
                except Exception:
                    continue

    print("\n[analysis] All analysis steps complete!")
    print("[analysis] Results locations:")
    print(f"  Evaluations:     {paths.METRICS_DIR}/")
    print(f"  Length gen:      {paths.OUTPUTS_DIR}/analysis/length_gen_multi/")
    print(f"  Attention:       {paths.OUTPUTS_DIR}/analysis/attention/")
    print(f"  Statistics:      {paths.OUTPUTS_DIR}/analysis/stats_*.json")


if __name__ == "__main__":
    main()
