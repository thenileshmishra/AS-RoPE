"""Aggregate evaluation results into summary tables.

Usage:
    python -m pipeline.aggregate_results
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

METRICS_DIR = Path("outputs/metrics")


def load_paper_results() -> dict[str, dict]:
    """Load all paper eval results (greedy decoding only)."""
    results = defaultdict(lambda: defaultdict(list))
    
    for eval_dir in sorted(METRICS_DIR.glob("*_eval")):
        summary_path = eval_dir / "eval_summary.json"
        if not summary_path.exists():
            continue
        
        data = json.loads(summary_path.read_text())
        run_name = eval_dir.name.replace("_eval", "")
        
        # Skip legacy inconsistent runs and beam-5 legacy evals
        # We only want greedy evals from new runs
        if data.get("decode_mode") == "beam-5" and "_s4" not in run_name and "_correct" not in run_name:
            continue
            
        pe_type = data.get("pe_type", "unknown")
        
        # Determine language
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
    
    return results


def print_table(results: dict, lang: str, lang_name: str) -> None:
    """Print a formatted results table."""
    if lang not in results:
        return
    
    print(f"\n{'='*70}")
    print(f"{lang_name} Results")
    print(f"{'='*70}")
    print(f"{'PE Type':<20} {'Runs':>5} {'BLEU':>8} {'chrF':>8} {'TER':>8}")
    print("-"*70)
    
    for pe_type in sorted(results[lang].keys()):
        runs = results[lang][pe_type]
        bleu_vals = [r["bleu"] for r in runs if r["bleu"] is not None]
        chrf_vals = [r["chrf"] for r in runs if r["chrf"] is not None]
        ter_vals = [r["ter"] for r in runs if r["ter"] is not None]
        
        if bleu_vals:
            bleu_mean = sum(bleu_vals) / len(bleu_vals)
            chrf_mean = sum(chrf_vals) / len(chrf_vals)
            ter_mean = sum(ter_vals) / len(ter_vals)
            bleu_std = (sum((x - bleu_mean)**2 for x in bleu_vals) / (len(bleu_vals) - 1))**0.5 if len(bleu_vals) > 1 else 0
            print(f"{pe_type:<20} {len(runs):>5} {bleu_mean:>7.2f}±{bleu_std:<4.2f} {chrf_mean:>7.2f} {ter_mean:>7.2f}")
            
            # Print individual runs
            for r in runs:
                print(f"  {r['run']:<18} {r['bleu']:>7.2f} {r['chrf']:>7.2f} {r['ter']:>7.2f}")


def save_latex_table(results: dict, out_path: str = "outputs/analysis/paper_table.txt") -> None:
    """Generate LaTeX table code."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{BLEU scores by positional encoding method. Mean ± std over seeds.}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & En-De & Hi-En & Bn-En \\")
    lines.append(r"\midrule")
    
    pe_order = ["rope", "adaptiverope", "gatesonly", "phasesonly", "sinusoidal", "alibi"]
    pe_names = {
        "rope": "RoPE",
        "adaptiverope": "AdaptiveRoPE",
        "gatesonly": "GatesOnly",
        "phasesonly": "PhasesOnly", 
        "sinusoidal": "Sinusoidal",
        "alibi": "ALiBi",
    }
    
    for pe in pe_order:
        row = [pe_names.get(pe, pe)]
        for lang in ["de", "hi", "bn"]:
            runs = results.get(lang, {}).get(pe, [])
            bleu_vals = [r["bleu"] for r in runs if r["bleu"] is not None]
            if bleu_vals:
                mean = sum(bleu_vals) / len(bleu_vals)
                if len(bleu_vals) > 1:
                    std = (sum((x - mean)**2 for x in bleu_vals) / (len(bleu_vals) - 1))**0.5
                    row.append(f"${mean:.2f} \\pm {std:.2f}$")
                else:
                    row.append(f"${mean:.2f}$")
            else:
                row.append("—")
        lines.append(" & ".join(row) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    out.write_text("\n".join(lines))
    print(f"\nLaTeX table saved to {out}")


def main():
    results = load_paper_results()
    
    print_table(results, "de", "English-German (WMT14)")
    print_table(results, "hi", "Hindi-English (Samanantar)")
    print_table(results, "bn", "Bengali-English (Samanantar)")
    
    save_latex_table(results)


if __name__ == "__main__":
    main()
