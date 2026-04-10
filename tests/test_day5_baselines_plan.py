"""Tests for the Day 5 baseline orchestrator and results collector.

Verifies that:
- config is parsed correctly
- command plans are generated for all 4 baselines
- manifest is written in dry-run mode
- no training is executed in default mode
- results collector handles missing and present metrics
"""

import csv
import json
import tempfile
from pathlib import Path

from scripts.run_day5_baselines import build_plan, main as orchestrator_main, write_manifest
from scripts.collect_day5_results import collect, write_summary


def _make_config(tmp: Path) -> Path:
    """Write a minimal baseline config pointing to tmp paths."""
    config = {
        "description": "test config",
        "shared": {
            "tokenizer": "gpt2",
            "max_seq_len": 32,
            "batch_size": 4,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "num_steps": 5,
            "eval_every": 5,
            "seed": 42,
            "device": "cpu",
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 2,
            "max_new_tokens": 8,
        },
        "data": {
            "train_file": str(tmp / "train.tsv"),
            "val_file": str(tmp / "val.tsv"),
            "eval_tsv": str(tmp / "test.tsv"),
        },
        "baselines": [
            {"name": "rope", "positional_encoding": "rope"},
            {"name": "alibi", "positional_encoding": "alibi"},
            {"name": "ntk_scaled_rope", "positional_encoding": "ntk_scaled_rope"},
            {"name": "sinusoidal", "positional_encoding": "sinusoidal"},
        ],
        "output_root": str(tmp / "results"),
    }
    path = tmp / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def test_build_plan_generates_all_stages():
    """Plan has 3 stages (train, decode, eval) per baseline = 12 entries."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        plan = build_plan(config)

        assert len(plan) == 12  # 4 baselines * 3 stages

        # Each baseline has exactly train, decode, eval in order
        names = [e["name"] for e in plan]
        stages = [e["stage"] for e in plan]
        for i in range(0, 12, 3):
            assert stages[i] == "train"
            assert stages[i + 1] == "decode"
            assert stages[i + 2] == "eval"
            assert names[i] == names[i + 1] == names[i + 2]

        # All 4 baselines are present
        baseline_names = sorted(set(names))
        assert baseline_names == ["alibi", "ntk_scaled_rope", "rope", "sinusoidal"]


def test_build_plan_commands_contain_pe_flag():
    """Every train command includes the correct --positional-encoding value."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        plan = build_plan(config)

        for entry in plan:
            if entry["stage"] == "train":
                cmd = entry["command"]
                idx = cmd.index("--positional-encoding")
                pe_val = cmd[idx + 1]
                assert pe_val == entry["name"]


def test_write_manifest():
    """Manifest JSON is written with expected fields."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        plan = build_plan(config)
        output_root = Path(config["output_root"])

        manifest_path = write_manifest(config, plan, output_root)
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "timestamp" in manifest
        assert "commit" in manifest
        assert "config" in manifest
        assert "commands" in manifest
        assert len(manifest["commands"]) == 12


def test_dry_run_does_not_execute():
    """Default orchestrator mode writes manifest but does not run commands."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        output_root = Path(config["output_root"])

        result = orchestrator_main(["--config", str(config_path)])

        assert result["executed"] is False
        assert result["plan_size"] == 12
        assert result["run_results"] is None
        assert Path(result["manifest_path"]).exists()

        # No training artifacts should exist
        for baseline in config["baselines"]:
            ckpt = output_root / baseline["name"] / "best.pt"
            assert not ckpt.exists(), f"Checkpoint should not exist in dry-run: {ckpt}"


def test_collect_handles_missing_results():
    """Collector reports status='missing' when eval outputs don't exist yet."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        rows = collect(config_path)

        assert len(rows) == 4
        for row in rows:
            assert row["status"] == "missing"
            assert row["bleu"] is None
            assert row["chrf"] is None


def test_collect_reads_existing_metrics():
    """Collector reads metrics.json when present."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        output_root = Path(config["output_root"])

        # Simulate one completed baseline
        eval_dir = output_root / "rope" / "eval"
        eval_dir.mkdir(parents=True)
        (eval_dir / "metrics.json").write_text(
            json.dumps({"bleu": 5.3, "chrf": 22.1}), encoding="utf-8"
        )

        rows = collect(config_path)
        rope_row = [r for r in rows if r["run_name"] == "rope"][0]
        assert rope_row["status"] == "ok"
        assert rope_row["bleu"] == 5.3
        assert rope_row["chrf"] == 22.1

        # Others still missing
        others = [r for r in rows if r["run_name"] != "rope"]
        assert all(r["status"] == "missing" for r in others)


def test_write_summary_csv_and_json():
    """Summary writer creates valid CSV and JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rows = [
            {"run_name": "rope", "positional_encoding": "rope",
             "bleu": 5.0, "chrf": 20.0, "checkpoint_path": "x", "status": "ok"},
            {"run_name": "alibi", "positional_encoding": "alibi",
             "bleu": None, "chrf": None, "checkpoint_path": "y", "status": "missing"},
        ]
        csv_path, json_path = write_summary(rows, tmp)

        assert csv_path.exists()
        assert json_path.exists()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 2
        assert csv_rows[0]["run_name"] == "rope"
        assert csv_rows[0]["bleu"] == "5.0"

        saved = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(saved) == 2


if __name__ == "__main__":
    test_build_plan_generates_all_stages()
    test_build_plan_commands_contain_pe_flag()
    test_write_manifest()
    test_dry_run_does_not_execute()
    test_collect_handles_missing_results()
    test_collect_reads_existing_metrics()
    test_write_summary_csv_and_json()
    print("All Day 5 baselines plan tests passed.")
