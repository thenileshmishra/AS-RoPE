"""Day 5 baseline orchestrator.

Reads configs/day5_baselines.json and builds the full command plan
for train -> decode -> eval for each positional encoding variant.

Modes:
  default (dry-run) : prints planned commands and writes manifest.json
  --execute         : actually runs each command via subprocess
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def build_plan(config: dict) -> list[dict]:
    """Build the full command plan from config.

    Returns a list of dicts with keys: name, stage, command, output_path.
    """
    shared = config["shared"]
    data = config["data"]
    output_root = Path(config["output_root"])
    plan: list[dict] = []

    for baseline in config["baselines"]:
        name = baseline["name"]
        pe = baseline["positional_encoding"]
        run_dir = output_root / name
        eval_dir = run_dir / "eval"

        # Stage 1: train
        train_cmd = [
            sys.executable, "-m", "src.train_mt",
            "--train-file", data["train_file"],
            "--val-file", data["val_file"],
            "--tokenizer", shared["tokenizer"],
            "--positional-encoding", pe,
            "--max-seq-len", str(shared["max_seq_len"]),
            "--batch-size", str(shared["batch_size"]),
            "--learning-rate", str(shared["learning_rate"]),
            "--weight-decay", str(shared["weight_decay"]),
            "--num-steps", str(shared["num_steps"]),
            "--eval-every", str(shared["eval_every"]),
            "--save-dir", str(run_dir),
            "--seed", str(shared["seed"]),
            "--device", shared["device"],
            "--d-model", str(shared["d_model"]),
            "--n-layers", str(shared["n_layers"]),
            "--n-heads", str(shared["n_heads"]),
        ]
        plan.append({
            "name": name,
            "stage": "train",
            "command": train_cmd,
            "output_path": str(run_dir / "best.pt"),
        })

        # Stage 2: decode
        decode_cmd = [
            sys.executable, "-m", "src.decode_mt",
            "--checkpoint", str(run_dir / "best.pt"),
            "--input-tsv", data["eval_tsv"],
            "--output-pred", str(eval_dir / "pred.txt"),
            "--output-ref", str(eval_dir / "ref.txt"),
            "--max-new-tokens", str(shared["max_new_tokens"]),
            "--device", shared["device"],
        ]
        plan.append({
            "name": name,
            "stage": "decode",
            "command": decode_cmd,
            "output_path": str(eval_dir / "pred.txt"),
        })

        # Stage 3: eval
        eval_cmd = [
            sys.executable, "-m", "src.eval_mt",
            "--pred-file", str(eval_dir / "pred.txt"),
            "--ref-file", str(eval_dir / "ref.txt"),
            "--output-json", str(eval_dir / "metrics.json"),
        ]
        plan.append({
            "name": name,
            "stage": "eval",
            "command": eval_cmd,
            "output_path": str(eval_dir / "metrics.json"),
        })

    return plan


def write_manifest(config: dict, plan: list[dict], output_root: Path) -> Path:
    """Write a manifest JSON for reproducibility."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _git_hash(),
        "config": config,
        "commands": [
            {
                "name": entry["name"],
                "stage": entry["stage"],
                "command": " ".join(entry["command"]),
                "output_path": entry["output_path"],
            }
            for entry in plan
        ],
    }
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def execute_plan(plan: list[dict]) -> list[dict]:
    """Run each command. Returns a list of result dicts."""
    results: list[dict] = []
    for entry in plan:
        label = f"{entry['name']}/{entry['stage']}"
        print(f"\n{'='*60}")
        print(f"[execute] {label}")
        print(f"[execute] {' '.join(entry['command'])}")
        print(f"{'='*60}")
        try:
            subprocess.check_call(entry["command"])
            results.append({"name": entry["name"], "stage": entry["stage"], "status": "ok"})
            print(f"[execute] {label} -> OK")
        except subprocess.CalledProcessError as e:
            results.append({
                "name": entry["name"],
                "stage": entry["stage"],
                "status": "failed",
                "returncode": e.returncode,
            })
            print(f"[execute] {label} -> FAILED (rc={e.returncode})")
    return results


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Day 5 baseline orchestrator")
    parser.add_argument(
        "--config",
        default="configs/day5_baselines.json",
        help="Path to baseline config JSON",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run commands (default: dry-run only)",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    output_root = Path(config["output_root"])
    plan = build_plan(config)
    manifest_path = write_manifest(config, plan, output_root)

    print(f"[baselines] config   : {config_path}")
    print(f"[baselines] manifest : {manifest_path}")
    print(f"[baselines] baselines: {[b['name'] for b in config['baselines']]}")
    print(f"[baselines] mode     : {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print()

    for entry in plan:
        tag = f"[{entry['name']}/{entry['stage']}]"
        print(f"{tag} {' '.join(entry['command'])}")

    run_results = None
    if args.execute:
        run_results = execute_plan(plan)
        failures = [r for r in run_results if r["status"] != "ok"]
        print(f"\n[baselines] completed: {len(run_results) - len(failures)} ok, {len(failures)} failed")
        if failures:
            for f in failures:
                print(f"  FAILED: {f['name']}/{f['stage']}")

    return {
        "manifest_path": str(manifest_path),
        "plan_size": len(plan),
        "executed": args.execute,
        "run_results": run_results,
    }


if __name__ == "__main__":
    main()
