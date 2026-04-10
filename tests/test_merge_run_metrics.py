"""Tests for scripts/merge_run_metrics.py."""

import csv
import json
import tempfile
from pathlib import Path

from scripts.merge_run_metrics import MERGED_COLUMNS, merge, write_merged


def _make_config(tmp: Path) -> Path:
    config = {
        "shared": {"seed": 42},
        "data": {"train_file": "x", "val_file": "y", "eval_tsv": "z"},
        "baselines": [
            {"name": "rope", "positional_encoding": "rope"},
            {"name": "alibi", "positional_encoding": "alibi"},
            {"name": "ntk_scaled_rope", "positional_encoding": "ntk_scaled_rope"},
            {"name": "sinusoidal", "positional_encoding": "sinusoidal"},
        ],
        "output_root": str(tmp / "results"),
    }
    p = tmp / "config.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return p


def _write_run_summary(run_dir: Path, **fields):
    run_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "best_val_loss": 4.2,
        "best_step": 100,
        "final_val_loss": 4.5,
        "total_steps": 200,
        "total_train_time_sec": 60.5,
        "avg_step_time_sec": 0.3,
        "total_tokens_seen": 50000,
        "positional_encoding": "rope",
        "seed": 42,
    }
    base.update(fields)
    (run_dir / "run_summary.json").write_text(json.dumps(base), encoding="utf-8")


def _write_eval_metrics(run_dir: Path, bleu: float, chrf: float):
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps({"bleu": bleu, "chrf": chrf, "num_samples": 8}), encoding="utf-8"
    )


def test_merge_all_present():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        output_root = tmp / "results"

        for i, name in enumerate(["rope", "alibi", "ntk_scaled_rope", "sinusoidal"]):
            _write_run_summary(output_root / name, positional_encoding=name)
            _write_eval_metrics(output_root / name, bleu=5.0 + i, chrf=20.0 + i)

        rows = merge(config_path)
        assert len(rows) == 4
        assert all(r["status"] == "ok" for r in rows)
        assert rows[0]["bleu"] == 5.0
        assert rows[1]["chrf"] == 21.0
        assert rows[0]["best_val_loss"] == 4.2


def test_merge_missing_files_graceful():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        rows = merge(config_path)
        assert len(rows) == 4
        for row in rows:
            assert row["status"] != "ok"
            assert "missing" in row["status"]
            assert row["bleu"] is None
            assert row["best_val_loss"] is None


def test_merge_partial_train_only():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        output_root = tmp / "results"

        _write_run_summary(output_root / "rope", positional_encoding="rope")
        # no eval metrics for rope

        rows = merge(config_path)
        rope_row = [r for r in rows if r["run_name"] == "rope"][0]
        assert rope_row["best_val_loss"] == 4.2
        assert rope_row["bleu"] is None
        assert "eval_missing" in rope_row["status"]


def test_write_merged_csv_and_json():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        output_root = tmp / "results"
        for i, name in enumerate(["rope", "alibi", "ntk_scaled_rope", "sinusoidal"]):
            _write_run_summary(output_root / name, positional_encoding=name)
            _write_eval_metrics(output_root / name, bleu=5.0 + i, chrf=20.0 + i)

        rows = merge(config_path)
        csv_path, json_path = write_merged(rows, output_root)

        assert csv_path.exists()
        assert json_path.exists()

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)
        assert len(csv_rows) == 4
        # Required columns are all present
        for col in MERGED_COLUMNS:
            assert col in csv_rows[0]

        json_rows = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(json_rows) == 4
        assert json_rows[0]["run_name"] == "rope"


def test_merge_handles_corrupt_json():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _make_config(tmp)
        output_root = tmp / "results"

        # Write a corrupt run_summary
        rope_dir = output_root / "rope"
        rope_dir.mkdir(parents=True, exist_ok=True)
        (rope_dir / "run_summary.json").write_text("not valid json", encoding="utf-8")

        rows = merge(config_path)
        rope_row = [r for r in rows if r["run_name"] == "rope"][0]
        assert "parse_error" in rope_row["status"]


if __name__ == "__main__":
    test_merge_all_present()
    test_merge_missing_files_graceful()
    test_merge_partial_train_only()
    test_write_merged_csv_and_json()
    test_merge_handles_corrupt_json()
    print("All merge_run_metrics tests passed.")
