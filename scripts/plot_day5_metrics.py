"""Generate Day 5 comparison plots from merged metrics + per-run logs.

Reads:
  - <output_root>/tables/runs_merged.json
  - <output_root>/<name>/metrics.jsonl  (per baseline)

Writes 5 PNGs to <output_root>/plots/:
  1. loss_curve_train.png      — train_loss vs step, all baselines
  2. loss_curve_val.png        — val_loss vs step, all baselines
  3. bleu_bar.png              — BLEU bar chart
  4. chrf_bar.png              — chrF bar chart
  5. quality_vs_time_scatter.png — BLEU vs total_train_time_sec scatter
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe in tests + CI
import matplotlib.pyplot as plt


def _load_metrics_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _baseline_metrics(output_root: Path, name: str) -> list[dict]:
    return _load_metrics_jsonl(output_root / name / "metrics.jsonl")


def plot_loss_curves(
    per_run_metrics: dict[str, list[dict]],
    field: str,
    title: str,
    out_path: Path,
) -> Path:
    """Plot a curve of `field` (e.g. train_loss, val_loss) per run."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted_any = False
    for name, rows in per_run_metrics.items():
        steps = [r["step"] for r in rows if field in r and r[field] is not None]
        values = [r[field] for r in rows if field in r and r[field] is not None]
        if steps and values:
            ax.plot(steps, values, marker="o", label=name, markersize=3)
            plotted_any = True

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(field)
    if plotted_any:
        ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_metric_bar(
    rows: list[dict],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = [r["run_name"] for r in rows]
    values = [r.get(metric_key) if isinstance(r.get(metric_key), (int, float)) else 0.0 for r in rows]
    bars = ax.bar(names, values, color=["#3b82f6", "#ef4444", "#10b981", "#f59e0b"][: len(names)])
    ax.set_title(title)
    ax.set_xlabel("baseline (positional encoding)")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_quality_vs_time(rows: list[dict], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        bleu = r.get("bleu")
        t = r.get("total_train_time_sec")
        if isinstance(bleu, (int, float)) and isinstance(t, (int, float)):
            ax.scatter(t, bleu, s=80)
            ax.annotate(r["run_name"], (t, bleu), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_title("Translation quality vs training time")
    ax.set_xlabel("total_train_time_sec")
    ax.set_ylabel("BLEU")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def generate_plots(config_path: str | Path) -> dict:
    """Generate all 5 plots. Returns a dict of plot name -> path."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    merged_path = output_root / "tables" / "runs_merged.json"
    if not merged_path.exists():
        raise FileNotFoundError(
            f"Merged metrics not found at {merged_path}. "
            "Run scripts.merge_run_metrics first."
        )
    merged_rows = json.loads(merged_path.read_text(encoding="utf-8"))

    per_run_metrics = {
        b["name"]: _baseline_metrics(output_root, b["name"])
        for b in config["baselines"]
    }

    train_path = plot_loss_curves(
        per_run_metrics, "train_loss", "Training loss curves", plots_dir / "loss_curve_train.png"
    )
    val_path = plot_loss_curves(
        per_run_metrics, "val_loss", "Validation loss curves", plots_dir / "loss_curve_val.png"
    )
    bleu_path = plot_metric_bar(
        merged_rows, "bleu", "BLEU by positional encoding", "BLEU", plots_dir / "bleu_bar.png"
    )
    chrf_path = plot_metric_bar(
        merged_rows, "chrf", "chrF by positional encoding", "chrF", plots_dir / "chrf_bar.png"
    )
    scatter_path = plot_quality_vs_time(merged_rows, plots_dir / "quality_vs_time_scatter.png")

    return {
        "loss_curve_train": str(train_path),
        "loss_curve_val": str(val_path),
        "bleu_bar": str(bleu_path),
        "chrf_bar": str(chrf_path),
        "quality_vs_time_scatter": str(scatter_path),
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Generate Day 5 comparison plots.")
    parser.add_argument("--config", default="configs/day5_baselines.json")
    args = parser.parse_args(argv)

    plots = generate_plots(args.config)
    print("[plot] generated:")
    for name, path in plots.items():
        print(f"  {name}: {path}")
    return plots


if __name__ == "__main__":
    main()
