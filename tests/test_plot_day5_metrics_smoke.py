"""Smoke test for scripts/plot_day5_metrics.py.

Builds a synthetic Day 5 results layout (merged JSON + per-run metrics.jsonl)
in a temp directory and verifies all 5 plots are produced and non-empty.
"""

import json
import tempfile
from pathlib import Path

from scripts.plot_day5_metrics import generate_plots

EXPECTED_PLOTS = [
    "loss_curve_train.png",
    "loss_curve_val.png",
    "bleu_bar.png",
    "chrf_bar.png",
    "quality_vs_time_scatter.png",
]

BASELINES = [
    {"name": "rope", "positional_encoding": "rope"},
    {"name": "alibi", "positional_encoding": "alibi"},
    {"name": "ntk_scaled_rope", "positional_encoding": "ntk_scaled_rope"},
    {"name": "sinusoidal", "positional_encoding": "sinusoidal"},
]


def _setup_synthetic_results(tmp: Path) -> Path:
    """Create a synthetic Day 5 results layout. Returns config path."""
    output_root = tmp / "results"
    output_root.mkdir(parents=True, exist_ok=True)

    config = {
        "shared": {"seed": 42},
        "data": {"train_file": "x", "val_file": "y", "eval_tsv": "z"},
        "baselines": BASELINES,
        "output_root": str(output_root),
    }
    config_path = tmp / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # Per-run metrics.jsonl
    for i, b in enumerate(BASELINES):
        run_dir = output_root / b["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for step in range(1, 11):
            row = {
                "step": step,
                "train_loss": 5.0 - 0.1 * step + 0.05 * i,
                "lr": 3e-4,
                "wall_time_sec": 1.0,
                "elapsed_sec": float(step),
                "tokens_in_batch": 32,
                "cumulative_tokens": 32 * step,
            }
            if step % 5 == 0:
                row["val_loss"] = 4.8 - 0.1 * step + 0.05 * i
                row["val_ppl"] = 50.0
            rows.append(row)
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )

    # Merged JSON (in tables/)
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    merged = [
        {
            "run_name": b["name"],
            "positional_encoding": b["positional_encoding"],
            "bleu": 5.0 + i,
            "chrf": 20.0 + i,
            "best_val_loss": 4.0 - 0.1 * i,
            "best_step": 10,
            "total_train_time_sec": 60.0 + 5 * i,
            "avg_step_time_sec": 0.3,
            "total_tokens_seen": 320,
            "seed": 42,
            "status": "ok",
        }
        for i, b in enumerate(BASELINES)
    ]
    (tables_dir / "runs_merged.json").write_text(json.dumps(merged), encoding="utf-8")

    return config_path


def test_generate_plots_creates_all_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config_path = _setup_synthetic_results(tmp)
        plots = generate_plots(config_path)

        # All 5 plot keys returned
        assert set(plots.keys()) == {
            "loss_curve_train",
            "loss_curve_val",
            "bleu_bar",
            "chrf_bar",
            "quality_vs_time_scatter",
        }

        plots_dir = Path(plots["loss_curve_train"]).parent
        assert plots_dir.name == "plots"

        # All 5 files exist and are non-empty
        for fname in EXPECTED_PLOTS:
            p = plots_dir / fname
            assert p.exists(), f"plot missing: {p}"
            assert p.stat().st_size > 0, f"plot empty: {p}"


def test_generate_plots_missing_merged_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Create config but no merged file
        output_root = tmp / "results"
        output_root.mkdir(parents=True, exist_ok=True)
        config = {
            "shared": {"seed": 42},
            "data": {},
            "baselines": BASELINES,
            "output_root": str(output_root),
        }
        config_path = tmp / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        try:
            generate_plots(config_path)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "runs_merged.json" in str(e)


if __name__ == "__main__":
    test_generate_plots_creates_all_files()
    test_generate_plots_missing_merged_raises()
    print("All plot_day5_metrics smoke tests passed.")
