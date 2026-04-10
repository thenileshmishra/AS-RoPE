"""One-command pretraining checks: init tracking + dataset stats + plan.

Runs the three pre-training scaffolding steps in order, without
launching any training. Useful as a sanity gate before committing
compute to the baseline matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.dataset_stats import compute_stats, write_stats
from scripts.init_experiment_tracking import init_experiment
from scripts.run_day5_baselines import build_plan, write_manifest


def run(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])

    # 1. Initialize tracking folders + manifest
    print("\n[1/3] init experiment tracking")
    manifest = init_experiment(config_path)
    manifest_path = output_root / "experiment_manifest.json"

    # 2. Dataset stats on train file
    print("\n[2/3] dataset stats")
    train_file = config["data"]["train_file"]
    stats_path = output_root / "artifacts" / "dataset_stats.json"
    stats: dict | None = None
    stats_status = "ok"
    try:
        stats = compute_stats(train_file)
        write_stats(stats, stats_path)
        print(f"[stats] {stats['num_samples']} samples in {train_file}")
    except FileNotFoundError as e:
        stats_status = f"missing: {e}"
        print(f"[stats] WARNING — {stats_status}")

    # 3. Baseline command plan + manifest
    print("\n[3/3] baseline command plan (dry-run)")
    plan = build_plan(config)
    plan_manifest_path = write_manifest(config, plan, output_root)

    print("\n" + "=" * 50)
    print("PRETRAINING CHECKLIST")
    print("=" * 50)
    print(f"  [{'OK' if manifest_path.exists() else '  '}] manifest generated     -> {manifest_path}")
    print(f"  [{'OK' if stats is not None else '  '}] dataset stats generated -> {stats_path}")
    print(f"  [{'OK' if plan else '  '}] command plan generated  -> {plan_manifest_path}")
    print(f"\n  baselines : {[b['name'] for b in config['baselines']]}")
    print(f"  plan size : {len(plan)} commands ({len(config['baselines'])} baselines x 3 stages)")

    return {
        "manifest_path": str(manifest_path),
        "stats_path": str(stats_path),
        "stats_status": stats_status,
        "plan_manifest_path": str(plan_manifest_path),
        "plan_size": len(plan),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Run all Day 5 pretraining checks.")
    parser.add_argument("--config", default="configs/day5_baselines.json")
    args = parser.parse_args(argv)
    return run(args.config)


if __name__ == "__main__":
    main()
