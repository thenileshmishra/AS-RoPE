"""Initialize Day 5 experiment tracking folder structure + manifest.

Creates results/day5_baselines/{runs,artifacts,plots,tables,logs} and
writes experiment_manifest.json — the single source of truth for what
gets compared, with what hyperparameters, on what data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT_SUBDIRS = ["runs", "artifacts", "plots", "tables", "logs"]


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


def init_experiment(config_path: str | Path) -> dict:
    """Create folder layout and write experiment_manifest.json. Returns the manifest."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    for sub in EXPERIMENT_SUBDIRS:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _git_hash(),
        "config_path": str(config_path),
        "output_root": str(output_root),
        "seed_policy": {
            "seed": config["shared"]["seed"],
            "policy": "single fixed seed shared across all baselines for fair comparison",
        },
        "shared_hyperparameters": config["shared"],
        "dataset_paths": config["data"],
        "baselines": config["baselines"],
        "decode_settings": {
            "max_new_tokens": config["shared"].get("max_new_tokens", 64),
            "device": config["shared"].get("device", "cpu"),
            "strategy": "greedy",
        },
        "subdirs": {sub: str(output_root / sub) for sub in EXPERIMENT_SUBDIRS},
    }

    manifest_path = output_root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Initialize Day 5 experiment tracking.")
    parser.add_argument("--config", default="configs/day5_baselines.json")
    args = parser.parse_args(argv)

    manifest = init_experiment(args.config)
    output_root = Path(manifest["output_root"])

    print(f"[init] manifest -> {output_root / 'experiment_manifest.json'}")
    for sub in EXPERIMENT_SUBDIRS:
        print(f"[init]   subdir: {output_root / sub}")
    print(f"[init] commit  : {manifest['commit']}")
    print(f"[init] seed    : {manifest['seed_policy']['seed']}")
    print(f"[init] baselines: {[b['name'] for b in manifest['baselines']]}")
    return manifest


if __name__ == "__main__":
    main()
