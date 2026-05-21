"""Statistical significance testing for MT evaluation results.

Compares AdaptiveRoPE against baselines using:
  - Paired t-test
  - Wilcoxon signed-rank test
  - Bootstrap 95% confidence intervals

Usage:
    python -m src.statistical_tests \
        --results-glob "outputs/metrics/*_de_eval/eval_summary.json" \
        --baseline-pe rope \
        --target-pe adaptiverope \
        --out-file outputs/analysis/stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_results(glob_pattern: str) -> dict[str, dict]:
    """Load eval_summary.json files matching pattern."""
    results = {}
    for path in Path(".").glob(glob_pattern):
        data = json.loads(path.read_text())
        pe_type = data.get("pe_type", "unknown")
        results.setdefault(pe_type, []).append(data)
    return results


def extract_metric(results_list: list[dict], metric_path: str) -> list[float]:
    """Extract a metric from a list of result dicts.

    metric_path: dot-separated path like 'overall.bleu' or 'by_src_length.4.bleu'
    """
    values = []
    for r in results_list:
        val = r
        for key in metric_path.split("."):
            if isinstance(val, list):
                val = val[int(key)]
            else:
                val = val.get(key, None)
            if val is None:
                break
        if val is not None:
            values.append(float(val))
    return values


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for mean difference (a - b)."""
    rng = np.random.default_rng(42)
    diffs = []
    n = min(len(a), len(b))
    for _ in range(n_bootstrap):
        idx_a = rng.integers(0, len(a), size=len(a))
        idx_b = rng.integers(0, len(b), size=len(b))
        diff = np.mean(a[idx_a]) - np.mean(b[idx_b])
        diffs.append(diff)
    diffs = np.array(diffs)
    alpha = 1 - ci
    lower = np.percentile(diffs, alpha / 2 * 100)
    upper = np.percentile(diffs, (1 - alpha / 2) * 100)
    return float(lower), float(upper)


def compare_methods(
    baseline_values: list[float],
    target_values: list[float],
    metric_name: str = "metric",
) -> dict:
    """Run statistical tests comparing target vs baseline."""
    a = np.array(target_values)
    b = np.array(baseline_values)

    # Paired t-test (requires same number of samples)
    if len(a) == len(b) and len(a) > 1:
        t_stat, t_pval = stats.ttest_rel(a, b)
    else:
        t_stat, t_pval = stats.ttest_ind(a, b)

    # Wilcoxon signed-rank test
    if len(a) == len(b) and len(a) > 1:
        try:
            w_stat, w_pval = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_pval = None, None
    else:
        w_stat, w_pval = None, None

    # Bootstrap CI for mean difference
    ci_lower, ci_upper = bootstrap_ci(a, b)

    return {
        "metric": metric_name,
        "baseline_n": len(b),
        "target_n": len(a),
        "baseline_mean": float(b.mean()),
        "baseline_std": float(b.std(ddof=1)),
        "target_mean": float(a.mean()),
        "target_std": float(a.std(ddof=1)),
        "mean_diff": float(a.mean() - b.mean()),
        "t_statistic": float(t_stat) if t_stat is not None else None,
        "t_pvalue": float(t_pval) if t_pval is not None else None,
        "wilcoxon_statistic": float(w_stat) if w_stat is not None else None,
        "wilcoxon_pvalue": float(w_pval) if w_pval is not None else None,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "significant_at_05": (t_pval is not None and t_pval < 0.05),
    }


def run_comparison(
    results: dict[str, list[dict]],
    baseline_pe: str,
    target_pe: str,
    metrics: list[str] | None = None,
) -> dict:
    """Compare target PE against baseline across multiple metrics."""
    if metrics is None:
        metrics = ["overall.bleu", "overall.chrf", "overall.ter"]

    baseline_results = results.get(baseline_pe, [])
    target_results = results.get(target_pe, [])

    if not baseline_results:
        raise ValueError(f"No results found for baseline pe_type={baseline_pe}")
    if not target_results:
        raise ValueError(f"No results found for target pe_type={target_pe}")

    output = {
        "baseline_pe": baseline_pe,
        "target_pe": target_pe,
        "baseline_runs": len(baseline_results),
        "target_runs": len(target_results),
        "comparisons": [],
    }

    for metric in metrics:
        baseline_vals = extract_metric(baseline_results, metric)
        target_vals = extract_metric(target_results, metric)
        if baseline_vals and target_vals:
            comp = compare_methods(baseline_vals, target_vals, metric_name=metric)
            output["comparisons"].append(comp)

    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Statistical significance testing")
    parser.add_argument("--results-glob", required=True,
                        help="Glob pattern to find eval_summary.json files")
    parser.add_argument("--baseline-pe", default="rope")
    parser.add_argument("--target-pe", default="adaptiverope")
    parser.add_argument("--metrics", nargs="+", default=["overall.bleu", "overall.chrf", "overall.ter"])
    parser.add_argument("--out-file", default="outputs/analysis/statistical_tests.json")
    args = parser.parse_args(argv)

    results = load_results(args.results_glob)
    print(f"[stats] Loaded results for PE types: {list(results.keys())}")

    comparison = run_comparison(
        results,
        baseline_pe=args.baseline_pe,
        target_pe=args.target_pe,
        metrics=args.metrics,
    )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2))

    print(f"[stats] Results saved to {out_path}")
    print("\n" + "=" * 60)
    print(f"Comparison: {args.target_pe} vs {args.baseline_pe}")
    print("=" * 60)
    for comp in comparison["comparisons"]:
        sig = "***" if comp.get("significant_at_05") else "n.s."
        print(f"\n  {comp['metric']}:")
        print(f"    Baseline:  {comp['baseline_mean']:.3f} ± {comp['baseline_std']:.3f}")
        print(f"    Target:    {comp['target_mean']:.3f} ± {comp['target_std']:.3f}")
        print(f"    Diff:      {comp['mean_diff']:+.3f}")
        print(f"    t-test:    p = {comp['t_pvalue']:.4f} {sig}")
        if comp.get("wilcoxon_pvalue") is not None:
            print(f"    Wilcoxon:  p = {comp['wilcoxon_pvalue']:.4f} {sig}")
        print(f"    Boot CI:   [{comp['bootstrap_ci_95'][0]:+.3f}, {comp['bootstrap_ci_95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
