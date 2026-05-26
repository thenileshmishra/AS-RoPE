#!/usr/bin/env python3
"""Parallel training launcher that fixes ALL publication issues.

Fixes applied:
  1. ALiBi baseline added (seq=128, batch=256)
  2. sinusoidal_de RETRAINED with correct params (seq=128, batch=256, steps=25K)
  3. Hi-En/Bn-En baseline training for consistency within each language
  4. All En-De jobs use NO gradient checkpointing (~30% speedup)
  5. Hi/Bn jobs run in parallel (up to 3 concurrent)

Strategy:
  - En-De jobs: sequential (1 at a time, compute-saturated at 98%)
  - Hi/Bn jobs: parallel (up to 3 at a time, smaller batches leave GPU headroom)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], desc: str) -> subprocess.Popen:
    print(f"\n[{'='*60}")
    print(f"[LAUNCH] {desc}")
    print(f"[{'='*60}")
    return subprocess.Popen(cmd)


def wait_for_processes(procs: list[subprocess.Popen], descs: list[str]) -> None:
    """Wait for all processes to complete."""
    while procs:
        for i in range(len(procs) - 1, -1, -1):
            ret = procs[i].poll()
            if ret is not None:
                if ret != 0:
                    print(f"[WARNING] {descs[i]} exited with code {ret}")
                else:
                    print(f"[DONE] {descs[i]}")
                procs.pop(i)
                descs.pop(i)
        if procs:
            time.sleep(10)


def is_complete(run_name: str) -> bool:
    return Path(f"outputs/logs/{run_name}/run_summary.json").exists()


def build_job(pe_type: str, run_name: str, train: str, val: str,
              seq: int, batch: int, steps: int, lr: float, seed: int) -> tuple[list[str], str]:
    """Build a training command."""
    if is_complete(run_name):
        return [], run_name  # skip

    cmd = [
        sys.executable, "-m", "pipeline.train_model",
        "--pe-type", pe_type,
        "--run-name", run_name,
        "--tokenized-train", train,
        "--tokenized-val", val,
        "--max-seq-len", str(seq),
        "--batch-size", str(batch),
        "--num-steps", str(steps),
        "--eval-every", "1000",
        "--learning-rate", str(lr),
        "--grad-accum", "1",
        "--use-bf16", "--use-compile",
        "--seed", str(seed),
        # NOTE: --use-checkpoint intentionally omitted for ~30% speedup
    ]
    return cmd, run_name


def main() -> None:
    # ═══════════════════════════════════════════════════════════════════════
    # En-De jobs (sequential, 1 at a time)
    # All use: seq=128, batch=256, steps=25K, lr=1e-3
    # ═══════════════════════════════════════════════════════════════════════
    en_de_jobs = []

    # Core baselines + ablations with 3 seeds
    for pe_type, seeds in [
        ("rope", [43, 44]),
        ("adaptiverope", [43, 44]),
        ("sinusoidal", [43, 44]),
        ("gatesonly", [42, 43, 44]),
        ("phasesonly", [42, 43, 44]),
    ]:
        for seed in seeds:
            run_name = f"{pe_type}_de_s{seed}"
            cmd, _ = build_job(
                pe_type, run_name,
                "processed_data_wmt14/tokenized/train_wmt14_en_de.pt",
                "processed_data_wmt14/tokenized/val_wmt14_en_de.pt",
                seq=128, batch=256, steps=25000, lr=1e-3, seed=seed,
            )
            if cmd:
                en_de_jobs.append((cmd, run_name))

    # FIX 1: ALiBi baseline (1 seed, En-De)
    cmd, _ = build_job(
        "alibi", "alibi_de_s42",
        "processed_data_wmt14/tokenized/train_wmt14_en_de.pt",
        "processed_data_wmt14/tokenized/val_wmt14_en_de.pt",
        seq=128, batch=256, steps=25000, lr=1e-3, seed=42,
    )
    if cmd:
        en_de_jobs.append((cmd, "alibi_de_s42"))

    # FIX 2: Retrain sinusoidal_de with CORRECT params (overwrite old inconsistent one)
    # We delete the old inconsistent checkpoint first
    old_sinusoidal = Path("outputs/checkpoints/sinusoidal_de")
    old_sinusoidal_logs = Path("outputs/logs/sinusoidal_de")
    if old_sinusoidal.exists() and not is_complete("sinusoidal_de"):
        # Only delete if we haven't already started the retrain
        pass  # We'll let the training overwrite it

    # Actually, we train a NEW run with explicit correct params and rename later
    cmd, _ = build_job(
        "sinusoidal", "sinusoidal_de_correct",
        "processed_data_wmt14/tokenized/train_wmt14_en_de.pt",
        "processed_data_wmt14/tokenized/val_wmt14_en_de.pt",
        seq=128, batch=256, steps=25000, lr=1e-3, seed=42,
    )
    if cmd:
        en_de_jobs.append((cmd, "sinusoidal_de_correct"))

    # ═══════════════════════════════════════════════════════════════════════
    # Hi-En jobs (parallel, up to 3 concurrent)
    # All use: seq=192, batch=64, steps=75K, lr=5e-4
    # FIX 7: Add rope + adaptiverope baselines for consistency
    # ═══════════════════════════════════════════════════════════════════════
    hi_jobs = []
    for pe_type, seeds in [
        ("rope", [42]),
        ("adaptiverope", [42]),
        ("gatesonly", [42]),
        ("phasesonly", [42]),
    ]:
        for seed in seeds:
            run_name = f"{pe_type}_hi_s{seed}"
            cmd, _ = build_job(
                pe_type, run_name,
                "processed_data/tokenized/train_1m_hi_en.pt",
                "processed_data/tokenized/val_hi_en.pt",
                seq=192, batch=64, steps=75000, lr=5e-4, seed=seed,
            )
            if cmd:
                hi_jobs.append((cmd, run_name))

    # ═══════════════════════════════════════════════════════════════════════
    # Bn-En jobs (parallel, up to 3 concurrent)
    # All use: seq=192, batch=64, steps=75K, lr=5e-4
    # FIX 7: Add rope + adaptiverope baselines for consistency
    # ═══════════════════════════════════════════════════════════════════════
    bn_jobs = []
    for pe_type, seeds in [
        ("rope", [42]),
        ("adaptiverope", [42]),
        ("sinusoidal", [42]),
        ("gatesonly", [42]),
        ("phasesonly", [42]),
    ]:
        for seed in seeds:
            run_name = f"{pe_type}_bn_s{seed}"
            cmd, _ = build_job(
                pe_type, run_name,
                "processed_data_bn/tokenized/train_bn_en.pt",
                "processed_data_bn/tokenized/val_bn_en.pt",
                seq=192, batch=64, steps=75000, lr=5e-4, seed=seed,
            )
            if cmd:
                bn_jobs.append((cmd, run_name))

    all_jobs = en_de_jobs + hi_jobs + bn_jobs
    print(f"[LAUNCH] Total jobs to run: {len(all_jobs)}")
    print(f"[LAUNCH]   En-De: {len(en_de_jobs)} (sequential, 1 at a time)")
    print(f"[LAUNCH]   Hi-En: {len(hi_jobs)} (parallel, up to 3 at a time)")
    print(f"[LAUNCH]   Bn-En: {len(bn_jobs)} (parallel, up to 3 at a time)")
    print(f"[LAUNCH] NOTE: Gradient checkpointing DISABLED for ~30%% speedup")

    # ═══════════════════════════════════════════════════════════════════════
    # Run En-De jobs sequentially
    # ═══════════════════════════════════════════════════════════════════════
    if en_de_jobs:
        print(f"\n[LAUNCH] >>> Starting En-De jobs ({len(en_de_jobs)} runs) <<<")
        for cmd, run_name in en_de_jobs:
            proc = run_cmd(cmd, run_name)
            proc.wait()
            if proc.returncode != 0:
                print(f"[WARNING] {run_name} failed with code {proc.returncode}")
            else:
                print(f"[DONE] {run_name}")

    # ═══════════════════════════════════════════════════════════════════════
    # Run Hi/Bn jobs in parallel (up to 3 concurrent)
    # ═══════════════════════════════════════════════════════════════════════
    small_jobs = hi_jobs + bn_jobs
    if small_jobs:
        print(f"\n[LAUNCH] >>> Starting Hi-En + Bn-En jobs ({len(small_jobs)} runs) <<<")
        max_concurrent = 3
        procs: list[subprocess.Popen] = []
        descs: list[str] = []

        for cmd, run_name in small_jobs:
            while len(procs) >= max_concurrent:
                for i in range(len(procs) - 1, -1, -1):
                    ret = procs[i].poll()
                    if ret is not None:
                        if ret != 0:
                            print(f"[WARNING] {descs[i]} exited with code {ret}")
                        else:
                            print(f"[DONE] {descs[i]}")
                        procs.pop(i)
                        descs.pop(i)
                if len(procs) >= max_concurrent:
                    time.sleep(10)

            proc = run_cmd(cmd, run_name)
            procs.append(proc)
            descs.append(run_name)

        wait_for_processes(procs, descs)

    print("\n[LAUNCH] All training jobs complete!")


if __name__ == "__main__":
    main()
