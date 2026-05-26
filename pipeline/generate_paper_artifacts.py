"""Generate all paper artifacts: tables, plots, and summary statistics.

Usage:
    python -m pipeline.generate_paper_artifacts
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS_DIR = Path("outputs/metrics")
OUT_DIR = Path("outputs/analysis/paper_artifacts")

PE_ORDER = ["rope", "adaptiverope", "gatesonly", "phasesonly", "sinusoidal", "alibi"]
PE_LABELS = {
    "rope": "RoPE",
    "adaptiverope": "AdaptiveRoPE",
    "gatesonly": "GatesOnly",
    "phasesonly": "PhasesOnly",
    "sinusoidal": "Sinusoidal",
    "alibi": "ALiBi",
}
LANG_ORDER = ["de", "hi", "bn"]
LANG_LABELS = {
    "de": "En-De",
    "hi": "Hi-En",
    "bn": "Bn-En",
}


def load_paper_results() -> dict:
    """Load all paper eval results."""
    results = defaultdict(lambda: defaultdict(list))
    pi_results = defaultdict(lambda: defaultdict(list))

    for eval_dir in sorted(METRICS_DIR.glob("*_eval")):
        summary_path = eval_dir / "eval_summary.json"
        if not summary_path.exists():
            continue

        data = json.loads(summary_path.read_text())
        run_name = eval_dir.name.replace("_eval", "")

        # Only include new runs with consistent hyperparameters
        if "_s4" not in run_name and "_correct" not in run_name:
            continue

        pe_type = data.get("pe_type", "unknown")

        if "_de" in run_name:
            lang = "de"
        elif "_hi" in run_name:
            lang = "hi"
        elif "_bn" in run_name:
            lang = "bn"
        else:
            continue

        metrics = data.get("overall", {})
        results[lang][pe_type].append({
            "run": run_name,
            "bleu": metrics.get("bleu"),
            "chrf": metrics.get("chrf"),
            "ter": metrics.get("ter"),
        })

    # Load PI results
    for pi_dir in sorted(METRICS_DIR.glob("*_pi*")):
        summary_path = pi_dir / "eval_summary.json"
        if not summary_path.exists():
            continue

        data = json.loads(summary_path.read_text())
        run_name = pi_dir.name.replace("_eval", "").split("_pi")[0]

        if "_s4" not in run_name and "_correct" not in run_name:
            continue

        pe_type = data.get("pe_type", "unknown")
        scale = data.get("pi_scale", 0)

        if "_de" in run_name:
            lang = "de"
        elif "_hi" in run_name:
            lang = "hi"
        elif "_bn" in run_name:
            lang = "bn"
        else:
            continue

        metrics = data.get("overall", {})
        pi_results[lang][scale].append({
            "run": run_name,
            "bleu": metrics.get("bleu"),
            "chrf": metrics.get("chrf"),
            "ter": metrics.get("ter"),
        })

    return dict(results), dict(pi_results)


def compute_stats(values: list[float]) -> dict:
    """Compute mean, std, min, max for a list of values."""
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(arr),
    }


def generate_bleu_barplot(results: dict, out_dir: Path) -> None:
    """Generate BLEU comparison bar plot across methods and languages."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, lang in zip(axes, LANG_ORDER):
        pe_types = []
        means = []
        stds = []

        for pe in PE_ORDER:
            runs = results.get(lang, {}).get(pe, [])
            bleu_vals = [r["bleu"] for r in runs if r["bleu"] is not None]
            if bleu_vals:
                pe_types.append(PE_LABELS.get(pe, pe))
                means.append(np.mean(bleu_vals))
                stds.append(np.std(bleu_vals, ddof=1) if len(bleu_vals) > 1 else 0)

        x = np.arange(len(pe_types))
        colors = [plt.cm.tab10(i) for i in range(len(pe_types))]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(pe_types, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel("BLEU" if lang == "de" else "")
        ax.set_title(LANG_LABELS[lang])
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(means) * 1.2 if means else 10)

    fig.suptitle("BLEU Comparison by Positional Encoding Method", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / "bleu_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[artifacts] Saved {out_dir / 'bleu_comparison.png'}")


def generate_pi_plot(pi_results: dict, out_dir: Path) -> None:
    """Generate Position Interpolation effect plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, lang in zip(axes, LANG_ORDER):
        scales = sorted([s for s in pi_results.get(lang, {}).keys() if s > 0])
        if not scales:
            ax.set_title(f"{LANG_LABELS[lang]} (no PI data)")
            continue

        means = []
        stds = []
        for scale in scales:
            runs = pi_results[lang][scale]
            bleu_vals = [r["bleu"] for r in runs if r["bleu"] is not None]
            if bleu_vals:
                means.append(np.mean(bleu_vals))
                stds.append(np.std(bleu_vals, ddof=1) if len(bleu_vals) > 1 else 0)

        # Also plot baseline (scale=1)
        baseline_runs = []
        for pe in ["rope"]:
            baseline_runs.extend(results.get(lang, {}).get(pe, []))
        baseline_bleu = [r["bleu"] for r in baseline_runs if r["bleu"] is not None]
        if baseline_bleu:
            scales = [1.0] + scales
            means = [np.mean(baseline_bleu)] + means
            stds = [np.std(baseline_bleu, ddof=1) if len(baseline_bleu) > 1 else 0] + stds

        ax.errorbar(scales, means, yerr=stds, marker='o', linewidth=2, markersize=8, capsize=4)
        ax.set_xlabel("PI Scale")
        ax.set_ylabel("BLEU" if lang == "de" else "")
        ax.set_title(LANG_LABELS[lang])
        ax.grid(alpha=0.3)
        ax.set_xticks(scales)

    fig.suptitle("Position Interpolation: BLEU vs Scale Factor", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / "pi_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[artifacts] Saved {out_dir / 'pi_comparison.png'}")


def generate_latex_table(results: dict, out_dir: Path) -> None:
    """Generate LaTeX table for main results."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{BLEU scores by positional encoding method (greedy decoding). Mean $\pm$ std over multiple seeds.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & En-De & Hi-En & Bn-En \\")
    lines.append(r"\midrule")

    for pe in PE_ORDER:
        row = [PE_LABELS.get(pe, pe)]
        for lang in LANG_ORDER:
            runs = results.get(lang, {}).get(pe, [])
            bleu_vals = [r["bleu"] for r in runs if r["bleu"] is not None]
            if bleu_vals:
                mean = np.mean(bleu_vals)
                if len(bleu_vals) > 1:
                    std = np.std(bleu_vals, ddof=1)
                    row.append(f"${mean:.2f} \\pm {std:.2f}$")
                else:
                    row.append(f"${mean:.2f}$")
            else:
                row.append("—")
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out_path = out_dir / "main_results_table.tex"
    out_path.write_text("\n".join(lines))
    print(f"[artifacts] Saved {out_path}")


def generate_summary_json(results: dict, pi_results: dict, out_dir: Path) -> None:
    """Generate a comprehensive summary JSON."""
    summary = {
        "methods": {},
        "by_language": {},
        "pi_effects": {},
    }

    for lang in LANG_ORDER:
        summary["by_language"][lang] = {}
        for pe in PE_ORDER:
            runs = results.get(lang, {}).get(pe, [])
            bleu_vals = [r["bleu"] for r in runs if r["bleu"] is not None]
            chrf_vals = [r["chrf"] for r in runs if r["chrf"] is not None]
            ter_vals = [r["ter"] for r in runs if r["ter"] is not None]
            if bleu_vals:
                summary["by_language"][lang][pe] = {
                    "bleu": compute_stats(bleu_vals),
                    "chrf": compute_stats(chrf_vals),
                    "ter": compute_stats(ter_vals),
                    "runs": [r["run"] for r in runs],
                }

    # PI effects
    for lang in LANG_ORDER:
        summary["pi_effects"][lang] = {}
        for scale in sorted(pi_results.get(lang, {}).keys()):
            if scale == 0:
                continue
            runs = pi_results[lang][scale]
            bleu_vals = [r["bleu"] for r in runs if r["bleu"] is not None]
            if bleu_vals:
                summary["pi_effects"][lang][f"scale_{scale}"] = compute_stats(bleu_vals)

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[artifacts] Saved {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global results, pi_results
    results, pi_results = load_paper_results()

    print(f"[artifacts] Loaded results for {len(results)} languages")
    for lang in LANG_ORDER:
        n_methods = len(results.get(lang, {}))
        print(f"  {lang}: {n_methods} methods")

    generate_bleu_barplot(results, OUT_DIR)
    generate_pi_plot(pi_results, OUT_DIR)
    generate_latex_table(results, OUT_DIR)
    generate_summary_json(results, pi_results, OUT_DIR)

    print("\n[artifacts] All paper artifacts generated!")
    print(f"  Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
