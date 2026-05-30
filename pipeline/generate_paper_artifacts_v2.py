"""
Generate comprehensive paper artifacts for top-tier conference submission.
"""

import json, os, math, statistics
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("outputs/analysis/paper_artifacts_v2", exist_ok=True)
OUT = "outputs/analysis/paper_artifacts_v2"

# ── Data loading ─────────────────────────────────────────────────────────────
PAPER_CPS = {
    "adaptiverope_de_s43", "adaptiverope_de_s44", "adaptiverope_hi_s42", "adaptiverope_bn_s42",
    "gatesonly_de_s42", "gatesonly_de_s43", "gatesonly_de_s44", "gatesonly_hi_s42", "gatesonly_bn_s42",
    "phasesonly_de_s42", "phasesonly_de_s43", "phasesonly_de_s44", "phasesonly_hi_s42", "phasesonly_bn_s42",
    "rope_de_s43", "rope_de_s44", "rope_hi_s42", "rope_bn_s42",
    "sinusoidal_de_correct", "sinusoidal_de_s43", "sinusoidal_de_s44", "sinusoidal_hi", "sinusoidal_bn_s42",
    "alibi_de_s42", "alibi_hi", "alibi_bn",
}

METRICS_DIR = "outputs/metrics"
ATTN_DIR = "outputs/analysis/attention"

results = defaultdict(lambda: defaultdict(list))
for cp in PAPER_CPS:
    d = cp + "_eval"
    path = os.path.join(METRICS_DIR, d, "eval_summary.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        data = json.load(f)
    parts = cp.split("_")
    if len(parts) == 2:
        # e.g. "sinusoidal_hi", "alibi_bn"
        method, lang = parts[0], parts[1]
    else:
        lang = parts[-2] if parts[-1].startswith("s") else parts[-1]
        method = "_".join(parts[:-2])
    o = data.get("overall", {})
    results[method][lang].append({
        "bleu": o.get("bleu", 0),
        "chrf": o.get("chrf", 0),
        "ter": o.get("ter", 0),
        "decode_sec": data.get("decode_sec", 0),
        "by_length": data.get("by_src_length", []),
        "n_params": data.get("n_params", 47131008),
    })

# Load length gen multi
with open("outputs/analysis/length_gen_multi/length_gen_multi_results.json") as f:
    lg_data = json.load(f)

# Load attention stats
attn_stats = defaultdict(lambda: defaultdict(list))
for cp in PAPER_CPS:
    path = os.path.join(ATTN_DIR, cp, "attention_analysis.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)
    parts = cp.split("_")
    if len(parts) == 2:
        # e.g. "sinusoidal_hi", "alibi_bn"
        method, lang = parts[0], parts[1]
    else:
        lang = parts[-2] if parts[-1].startswith("s") else parts[-1]
        method = "_".join(parts[:-2])
    entropies = [layer["mean_entropy"] for layer in d.get("per_layer", [])]
    sinks = [layer["mean_sink_ratio"] for layer in d.get("per_layer", [])]
    attn_stats[method][lang].append({
        "mean_entropy": sum(entropies)/len(entropies) if entropies else 0,
        "mean_sink": sum(sinks)/len(sinks) if sinks else 0,
    })

# ── Helper ───────────────────────────────────────────────────────────────────
def fmt_mean_std(vals, fmt=".2f"):
    if len(vals) == 1:
        return ("{0:" + fmt + "}").format(vals[0])
    m = statistics.mean(vals)
    s = statistics.stdev(vals)
    return ("{0:" + fmt + "}$\\pm${1:" + fmt + "}").format(m, s)

def get_vals(method, lang, key):
    return [e[key] for e in results[method].get(lang, [])]

# ── Table 1: Main Results (BLEU, chrF, TER, Speed) ──────────────────────────
methods_order = ["rope", "adaptiverope", "gatesonly", "phasesonly", "sinusoidal", "alibi"]
method_names = {
    "rope": "RoPE",
    "adaptiverope": "AdaptiveRoPE",
    "gatesonly": "GatesOnly",
    "phasesonly": "PhasesOnly",
    "sinusoidal": "Sinusoidal",
    "alibi": "ALiBi",
}

table1 = r"""\begin{table*}[t]
\centering
\small
\caption{Translation quality and inference speed across positional encoding methods and language pairs.
En--De reports mean~$\pm$~std over 2--3 seeds; Hi--En and Bn--En use one seed. Bold = best per column.
$\uparrow$/$\downarrow$ indicate preferred direction.}
\label{tab:main}
\begin{tabular}{l|ccc|ccc|ccc|c}
\toprule
& \multicolumn{3}{c|}{\textbf{En--De}} & \multicolumn{3}{c|}{\textbf{Hi--En}} & \multicolumn{3}{c|}{\textbf{Bn--En}} & \\
\textbf{Method} & BLEU$\uparrow$ & chrF$\uparrow$ & TER$\downarrow$ & BLEU$\uparrow$ & chrF$\uparrow$ & TER$\downarrow$ & BLEU$\uparrow$ & chrF$\uparrow$ & TER$\downarrow$ & Decode~s$\downarrow$ \\
\midrule
"""

for m in methods_order:
    if m not in results:
        continue
    row = [method_names[m]]
    for lang in ["de", "hi", "bn"]:
        if lang in results[m]:
            bleus = get_vals(m, lang, "bleu")
            chrfs = get_vals(m, lang, "chrf")
            ters = get_vals(m, lang, "ter")
            row.append(fmt_mean_std(bleus))
            row.append(fmt_mean_std(chrfs))
            row.append(fmt_mean_std(ters))
        else:
            row.extend(["—", "—", "—"])
    # speed: use de only
    if "de" in results[m]:
        secs = get_vals(m, "de", "decode_sec")
        row.append(fmt_mean_std(secs, ".1f"))
    else:
        row.append("—")
    table1 += " & ".join(row) + r" \\" + "\n"

table1 += r"""\bottomrule
\end{tabular}
\end{table*}
"""

with open(f"{OUT}/table_main.tex", "w") as f:
    f.write(table1)

# ── Table 2: Per-Source-Length BLEU (En-De, in-domain) ──────────────────────
table2 = r"""\begin{table}[t]
\centering
\small
\caption{BLEU by source sentence length bucket on En--De (greedy decoding, mean over seeds).
Training distribution spans buckets 1--50 words.}
\label{tab:perlength}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{1--10} & \textbf{11--20} & \textbf{21--30} & \textbf{31--50} & \textbf{51+} \\
\midrule
"""

length_ranges = ["1-10", "11-20", "21-30", "31-50", "51-10000"]
for m in methods_order:
    if "de" not in results[m]:
        continue
    row = [method_names[m]]
    for r in length_ranges:
        vals = []
        for e in results[m]["de"]:
            for bl in e.get("by_length", []):
                if bl["range"] == r:
                    vals.append(bl["bleu"])
                    break
        if vals:
            row.append(fmt_mean_std(vals))
        else:
            row.append("—")
    table2 += " & ".join(row) + r" \\" + "\n"

table2 += r"""\bottomrule
\end{tabular}
\end{table}
"""

with open(f"{OUT}/table_perlength.tex", "w") as f:
    f.write(table2)

# ── Table 3: Length Generalization ──────────────────────────────────────────
table3 = r"""\begin{table}[t]
\centering
\small
\caption{BLEU degradation on out-of-distribution source lengths (En--De, greedy).
Values are \% change relative to the 1--20 word bucket. OOD = outside training distribution.}
\label{tab:length}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{21--40} & \textbf{41--60} & \textbf{61--80} & \textbf{81--120} & \textbf{201--400} \\
\midrule
"""

lg_methods = ["RoPE", "AdaptiveRoPE", "GatesOnly", "PhasesOnly", "Sinusoidal", "ALiBi"]
lg_name_map = {"RoPE": "rope", "AdaptiveRoPE": "adaptiverope", "GatesOnly": "gatesonly",
               "PhasesOnly": "phasesonly", "Sinusoidal": "sinusoidal", "ALiBi": "alibi"}

ranges = ["21-40", "41-60", "61-80", "81-100", "101-150", "151-200", "201-400"]

table3_ranges = ["21-40", "41-60", "61-80", "81-100,101-150", "201-400"]

for m_disp in lg_methods:
    m = lg_name_map[m_disp]
    if m_disp not in lg_data["methods"]:
        continue
    row = [m_disp]
    res = {r["range"]: r for r in lg_data["methods"][m_disp]["results"]}
    # 21-40
    row.append(f"{res['21-40']['bleu_drop_pct']:+.1f}" if "21-40" in res else "—")
    # 41-60
    row.append(f"{res['41-60']['bleu_drop_pct']:+.1f}" if "41-60" in res else "—")
    # 61-80
    row.append(f"{res['61-80']['bleu_drop_pct']:+.1f}" if "61-80" in res else "—")
    # 81-120 combined
    if "81-100" in res and "101-150" in res:
        avg = (res["81-100"]["bleu_drop_pct"] + res["101-150"]["bleu_drop_pct"]) / 2
        row.append(f"{avg:+.1f}")
    else:
        row.append("—")
    # 201-400
    row.append(f"{res['201-400']['bleu_drop_pct']:+.1f}" if "201-400" in res else "—")
    table3 += " & ".join(row) + r" \\" + "\n"

table3 += r"""\bottomrule
\end{tabular}
\end{table}
"""

with open(f"{OUT}/table_length.tex", "w") as f:
    f.write(table3)

# ── Table 4: Attention Statistics ───────────────────────────────────────────
table4 = r"""\begin{table}[t]
\centering
\small
\caption{Attention entropy and sink ratio by positional encoding (En--De, mean across seeds).
Higher entropy = more uniform attention; higher sink ratio = stronger focus on initial tokens.}
\label{tab:attention}
\begin{tabular}{lcc}
\toprule
\textbf{Method} & \textbf{Mean Entropy}$\uparrow$ & \textbf{Mean Sink Ratio}$\uparrow$ \\
\midrule
"""

for m in methods_order:
    if "de" not in attn_stats[m]:
        continue
    ents = [e["mean_entropy"] for e in attn_stats[m]["de"]]
    sinks = [e["mean_sink"] for e in attn_stats[m]["de"]]
    table4 += f"{method_names[m]} & {fmt_mean_std(ents, '.3f')} & {fmt_mean_std(sinks, '.3f')} \\\\\n"

table4 += r"""\bottomrule
\end{tabular}
\end{table}
"""

with open(f"{OUT}/table_attention.tex", "w") as f:
    f.write(table4)

# ── Figure: Cross-lingual comparison bar chart ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

langs = ["de", "hi", "bn"]
lang_titles = ["En--De (3 seeds)", "Hi--En", "Bn--En"]

for ax, lang, title in zip(axes, langs, lang_titles):
    methods = ["rope", "adaptiverope", "gatesonly", "phasesonly"]
    if lang in ("de", "hi", "bn"):
        methods += ["sinusoidal", "alibi"]
    
    names = [method_names[m] for m in methods]
    means = []
    errs = []
    for m in methods:
        if lang in results[m]:
            bleus = get_vals(m, lang, "bleu")
            means.append(statistics.mean(bleus))
            errs.append(statistics.stdev(bleus) if len(bleus) > 1 else 0)
        else:
            means.append(0)
            errs.append(0)
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    bars = ax.bar(names, means, yerr=errs, capsize=3, color=colors[:len(names)], edgecolor='black', linewidth=0.5)
    ax.set_ylabel("BLEU")
    ax.set_title(title, fontweight='bold')
    ax.tick_params(axis='x', rotation=15)
    ax.set_ylim(bottom=min(means)*0.95 if means else 0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{mean:.2f}", ha='center', va='bottom', fontsize=8)

plt.suptitle("BLEU Comparison Across Language Pairs", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/crosslingual_comparison.png", dpi=200, bbox_inches='tight')
plt.close()

# ── Figure: Per-layer entropy comparison ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

layers = list(range(6))
for m in ["rope", "adaptiverope", "gatesonly", "phasesonly", "alibi", "sinusoidal"]:
    if m not in attn_stats or "de" not in attn_stats[m]:
        continue
    # Use first seed
    cp_name = None
    for cp in PAPER_CPS:
        if cp.startswith(m + "_de"):
            cp_name = cp
            break
    if not cp_name:
        continue
    path = os.path.join(ATTN_DIR, cp_name, "attention_analysis.json")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)
    entropies = [layer["mean_entropy"] for layer in d.get("per_layer", [])]
    ax.plot(layers, entropies, marker='o', label=method_names[m], linewidth=2)

ax.set_xlabel("Layer")
ax.set_ylabel("Mean Attention Entropy")
ax.set_title("Per-Layer Attention Entropy (En--De)", fontweight='bold')
ax.legend(loc='best')
ax.grid(alpha=0.3)
ax.set_xticks(layers)
plt.tight_layout()
plt.savefig(f"{OUT}/entropy_per_layer.png", dpi=200, bbox_inches='tight')
plt.close()

print(f"All artifacts generated in {OUT}/")
print(f"  - table_main.tex")
print(f"  - table_perlength.tex")
print(f"  - table_length.tex")
print(f"  - table_attention.tex")
print(f"  - crosslingual_comparison.png")
print(f"  - entropy_per_layer.png")
