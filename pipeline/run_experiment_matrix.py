"""Run the full experiment matrix for publication.

Trains all missing combinations of PE type, seed, and dataset.

Usage:
    python -m pipeline.run_experiment_matrix --dry-run    # preview only
    python -m pipeline.run_experiment_matrix               # execute all
    python -m pipeline.run_experiment_matrix --only-en-de  # main language only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline import paths


# ── Experiment configuration ────────────────────────────────────────────────

DATASETS = {
    "de": {
        "train": "processed_data_wmt14/tokenized/train_wmt14_en_de.pt",
        "val":   "processed_data_wmt14/tokenized/val_wmt14_en_de.pt",
        "test":  "raw_data/wmt14/test.tsv",
        "tokenizer": "Helsinki-NLP/opus-mt-en-de",
    },
    "hi": {
        "train": "processed_data/tokenized/train_1m_hi_en.pt",
        "val":   "processed_data/tokenized/val_hi_en.pt",
        "test":  "raw_data/samanantar/test.tsv",  # may not exist; eval uses val if missing
        "tokenizer": "Helsinki-NLP/opus-mt-en-de",  # same tokenizer used for hi
    },
    "bn": {
        "train": "processed_data_bn/tokenized/train_bn_en.pt",
        "val":   "processed_data_bn/tokenized/val_bn_en.pt",
        "test":  None,
        "tokenizer": "Helsinki-NLP/opus-mt-en-de",
    },
}

# Training hyperparameters (matching your existing successful runs)
TRAIN_CONFIG = {
    "batch_size": 512,
    "num_steps": 25_000,
    "eval_every": 1_000,
    "learning_rate": 1e-3,
    "grad_accum": 1,
    "use_checkpoint": True,
    "use_bf16": True,
    "use_compile": True,
    "max_seq_len": 256,
}

# Existing checkpoints (seed 42 assumed) — these will NOT be retrained
EXISTING_CHECKPOINTS = {
    # En-De
    ("de", "rope", 42):       "outputs/checkpoints/rope_de",
    ("de", "adaptiverope", 42): "outputs/checkpoints/asrope3_de",
    ("de", "sinusoidal", 42):  "outputs/checkpoints/sinusoidal_de",
    # Hi-En
    ("hi", "rope", 42):       "outputs/checkpoints/rope_hi",
    ("hi", "adaptiverope", 42): "outputs/checkpoints/asrope3_hi",
    ("hi", "sinusoidal", 42):  "outputs/checkpoints/sinusoidal_hi",
    # Bn-En
    ("bn", "rope", 42):       "outputs/checkpoints/rope_bn",
    ("bn", "adaptiverope", 42): "outputs/checkpoints/asrope3_bn",
}


def build_jobs(only_en_de: bool = False, seeds: tuple[int, ...] = (42, 43, 44)) -> list[dict]:
    """Build list of training jobs that need to run."""
    jobs = []
    langs = ["de"] if only_en_de else ["de", "hi", "bn"]

    for lang in langs:
        ds = DATASETS[lang]
        # For En-De: 3 seeds for all main methods
        # For Hi/Bn: 1 seed (42) for new methods only
        lang_seeds = seeds if lang == "de" else (42,)

        for pe_type in ["rope", "adaptiverope", "sinusoidal", "gatesonly", "phasesonly"]:
            for seed in lang_seeds:
                key = (lang, pe_type, seed)
                run_name = f"{pe_type}_{lang}_s{seed}"
                ckpt_dir = paths.CHECKPOINT_DIR / run_name

                # Skip if already exists and has best.pt
                if key in EXISTING_CHECKPOINTS or (ckpt_dir / "best.pt").exists():
                    continue

                jobs.append({
                    "run_name": run_name,
                    "pe_type": pe_type,
                    "seed": seed,
                    "lang": lang,
                    "train": ds["train"],
                    "val": ds["val"],
                    "tokenizer": ds["tokenizer"],
                })

    return jobs


def run_job(job: dict, dry_run: bool = False) -> None:
    """Launch a single training job."""
    cmd = [
        sys.executable, "-m", "pipeline.train_model",
        "--pe-type", job["pe_type"],
        "--run-name", job["run_name"],
        "--tokenized-train", job["train"],
        "--tokenized-val", job["val"],
        "--seed", str(job["seed"]),
        "--batch-size", str(TRAIN_CONFIG["batch_size"]),
        "--num-steps", str(TRAIN_CONFIG["num_steps"]),
        "--eval-every", str(TRAIN_CONFIG["eval_every"]),
        "--learning-rate", str(TRAIN_CONFIG["learning_rate"]),
        "--grad-accum", str(TRAIN_CONFIG["grad_accum"]),
        "--max-seq-len", str(TRAIN_CONFIG["max_seq_len"]),
    ]
    if TRAIN_CONFIG["use_checkpoint"]:
        cmd.append("--use-checkpoint")
    if TRAIN_CONFIG["use_bf16"]:
        cmd.append("--use-bf16")
    if TRAIN_CONFIG["use_compile"]:
        cmd.append("--use-compile")

    print(f"\n[matrix] {'(dry-run) ' if dry_run else ''}Running: {job['run_name']}")
    print(f"[matrix]   cmd: {' '.join(cmd)}")

    if dry_run:
        return

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[matrix] WARNING: {job['run_name']} failed with exit code {result.returncode}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run experiment matrix")
    parser.add_argument("--dry-run", action="store_true", help="Preview jobs without running")
    parser.add_argument("--only-en-de", action="store_true", help="Only English-German")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                        help="Seeds to use for En-De")
    args = parser.parse_args(argv)

    jobs = build_jobs(only_en_de=args.only_en_de, seeds=tuple(args.seeds))

    print(f"[matrix] Found {len(jobs)} jobs to run")
    print(f"[matrix] {'='*60}")
    for j in jobs:
        print(f"  {j['run_name']:40s}  train={j['train']}")
    print(f"[matrix] {'='*60}")

    if args.dry_run:
        print("[matrix] Dry run complete. Use without --dry-run to execute.")
        return

    for idx, job in enumerate(jobs, 1):
        print(f"\n[matrix] Job {idx}/{len(jobs)}")
        run_job(job, dry_run=False)

    print(f"\n[matrix] All {len(jobs)} jobs complete.")


if __name__ == "__main__":
    main()
