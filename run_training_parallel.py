#!/usr/bin/env python3
"""Parallel training launcher that maximizes GPU utilization.

Strategy:
  - En-De jobs: sequential (1 at a time) because batch=256 saturates GPU compute at 98%
  - Hi/Bn jobs: parallel (up to 3 at a time) because batch=64 leaves GPU underutilized
  - ALL jobs: NO gradient checkpointing for ~30% speedup (fits in 80GB A100)

This is faster than naive sequential training because:
  1. No checkpointing = ~30% faster per job
  2. Hi/Bn parallelized = multiple small jobs overlap
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
        # NOTE: --use-checkpoint is intentionally omitted for ~30% speedup
    ]
    return cmd, run_name


def main() -> None:
    # ═══════════════════════════════════════════════════════════════════════
    # Job definitions
    # ═══════════════════════════════════════════════════════════════════════

    en_de_jobs = []
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

    hi_jobs = []
    for pe_type, seeds in [
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

    bn_jobs = []
    for pe_type, seeds in [
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
    print(f"[LAUNCH] NOTE: En-De jobs saturate GPU compute (98%%), so running")
    print(f"[LAUNCH]       2 En-De jobs in parallel would make BOTH slower")
    print(f"[LAUNCH]       with ZERO net benefit. Sequential is optimal.")

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
            # Wait if we've hit the concurrency limit
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

        # Wait for remaining processes
        wait_for_processes(procs, descs)

    print("\n[LAUNCH] All training jobs complete!")


if __name__ == "__main__":
    main()
