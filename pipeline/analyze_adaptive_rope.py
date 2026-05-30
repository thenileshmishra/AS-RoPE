"""Comprehensive statistical analysis of AdaptiveRoPE parameters and results.

Produces:
  1. Bootstrap significance tests for all language pairs
  2. Gate/phase parameter visualizations from AdaptiveRoPE checkpoints
  3. Cross-lingual comparison of learned parameters
  4. Summary statistics table

Usage:
    python -m pipeline.analyze_adaptive_rope
    python -m pipeline.analyze_adaptive_rope --output-dir outputs/analysis
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

# Try to import matplotlib; if unavailable, skip plotting
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Configuration ───────────────────────────────────────────────────────────

CHECKPOINTS = {
    "adaptiverope_de_s42": "outputs/checkpoints/adaptiverope_de_s42/best.pt",
    "adaptiverope_de_s43": "outputs/checkpoints/adaptiverope_de_s43/best.pt",
    "adaptiverope_de_s44": "outputs/checkpoints/adaptiverope_de_s44/best.pt",
    "adaptiverope_hi_s42": "outputs/checkpoints/adaptiverope_hi_s42/best.pt",
    "adaptiverope_bn_s42": "outputs/checkpoints/adaptiverope_bn_s42/best.pt",
}

PREDICTIONS = {
    "en-de": {
        "RoPE": [
            "outputs/metrics/rope_de_eval/predictions.tsv",
            "outputs/metrics/rope_de_s43_eval/predictions.tsv",
            "outputs/metrics/rope_de_s44_eval/predictions.tsv",
        ],
        "AdaptiveRoPE": [
            "outputs/metrics/adaptiverope_de_s43_eval/predictions.tsv",
            "outputs/metrics/adaptiverope_de_s44_eval/predictions.tsv",
        ],
        "GatesOnly": [
            "outputs/metrics/gatesonly_de_s42_eval/predictions.tsv",
            "outputs/metrics/gatesonly_de_s43_eval/predictions.tsv",
            "outputs/metrics/gatesonly_de_s44_eval/predictions.tsv",
        ],
        "PhasesOnly": [
            "outputs/metrics/phasesonly_de_s42_eval/predictions.tsv",
            "outputs/metrics/phasesonly_de_s43_eval/predictions.tsv",
            "outputs/metrics/phasesonly_de_s44_eval/predictions.tsv",
        ],
        "ALiBi": [
            "outputs/metrics/alibi_de_s42_eval/predictions.tsv",
        ],
    },
    "hi-en": {
        "RoPE": ["outputs/metrics/rope_hi_s42_eval/predictions.tsv"],
        "AdaptiveRoPE": ["outputs/metrics/adaptiverope_hi_s42_eval/predictions.tsv"],
        "GatesOnly": ["outputs/metrics/gatesonly_hi_s42_eval/predictions.tsv"],
        "PhasesOnly": ["outputs/metrics/phasesonly_hi_s42_eval/predictions.tsv"],
        "ALiBi": ["outputs/metrics/alibi_hi_eval/predictions.tsv"],
    },
    "bn-en": {
        "RoPE": ["outputs/metrics/rope_bn_s42_eval/predictions.tsv"],
        "AdaptiveRoPE": ["outputs/metrics/adaptiverope_bn_s42_eval/predictions.tsv"],
        "GatesOnly": ["outputs/metrics/gatesonly_bn_s42_eval/predictions.tsv"],
        "PhasesOnly": ["outputs/metrics/phasesonly_bn_s42_eval/predictions.tsv"],
        "ALiBi": ["outputs/metrics/alibi_bn_eval/predictions.tsv"],
    },
}

N_BOOTSTRAP = 200
RANDOM_SEED = 42


# ── Bootstrap significance testing ──────────────────────────────────────────

def _load_predictions(tsv_path: Path) -> tuple[list[str], list[str]]:
    sources, refs, hyps = [], [], []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            sources.append(parts[0])
            refs.append(parts[1])
            hyps.append(parts[2])
    return refs, hyps


def _sentence_bleu(refs: list[str], hyps: list[str]) -> list[float]:
    """Compute corpus BLEU and return sentence-level approximate scores."""
    import sacrebleu

    # For bootstrap we use corpus_bleu on resampled sets
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


def bootstrap_bleu(refs: list[str], hyps: list[str], n: int = N_BOOTSTRAP, seed: int = RANDOM_SEED) -> dict:
    """Bootstrap resample sentences and compute BLEU distribution."""
    rng = random.Random(seed)
    m = len(refs)
    scores = []
    for _ in range(n):
        idxs = [rng.randint(0, m - 1) for _ in range(m)]
        r_samp = [refs[i] for i in idxs]
        h_samp = [hyps[i] for i in idxs]
        scores.append(_sentence_bleu(r_samp, h_samp))
    scores = np.array(scores)
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_low": float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
    }


def pairwise_bootstrap_test(refs1, hyps1, refs2, hyps2, n=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Paired bootstrap test: how often method1 > method2."""
    rng = random.Random(seed)
    m = len(refs1)
    assert m == len(refs2)
    wins = 0
    diffs = []
    for _ in range(n):
        idxs = [rng.randint(0, m - 1) for _ in range(m)]
        r1 = [refs1[i] for i in idxs]
        h1 = [hyps1[i] for i in idxs]
        r2 = [refs2[i] for i in idxs]
        h2 = [hyps2[i] for i in idxs]
        s1 = _sentence_bleu(r1, h1)
        s2 = _sentence_bleu(r2, h2)
        diffs.append(s1 - s2)
        if s1 > s2:
            wins += 1
    diffs = np.array(diffs)
    return {
        "p_win": wins / n,
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


# ── AdaptiveRoPE parameter extraction ───────────────────────────────────────

def extract_adaptive_rope_params(checkpoint_path: str):
    """Extract gate and phase parameters from an AdaptiveRoPE checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    params = {
        "encoder_gates_q": [],
        "encoder_gates_k": [],
        "encoder_phases_q": [],
        "encoder_phases_k": [],
        "decoder_gates_q": [],
        "decoder_gates_k": [],
        "decoder_phases_q": [],
        "decoder_phases_k": [],
    }

    for k, v in state.items():
        if "encoder" in k and "attn.pe.gates_q" in k:
            params["encoder_gates_q"].append(v.numpy())
        elif "encoder" in k and "attn.pe.gates_k" in k:
            params["encoder_gates_k"].append(v.numpy())
        elif "encoder" in k and "attn.pe.phase_q" in k:
            params["encoder_phases_q"].append(v.numpy())
        elif "encoder" in k and "attn.pe.phase_k" in k:
            params["encoder_phases_k"].append(v.numpy())
        elif "decoder" in k and "self_attn.pe.gates_q" in k:
            params["decoder_gates_q"].append(v.numpy())
        elif "decoder" in k and "self_attn.pe.gates_k" in k:
            params["decoder_gates_k"].append(v.numpy())
        elif "decoder" in k and "self_attn.pe.phase_q" in k:
            params["decoder_phases_q"].append(v.numpy())
        elif "decoder" in k and "self_attn.pe.phase_k" in k:
            params["decoder_phases_k"].append(v.numpy())

    # Stack into arrays: (n_layers, n_heads, n_freqs)
    for key in params:
        if params[key]:
            params[key] = np.stack(params[key], axis=0)
        else:
            params[key] = None

    return params


def param_stats(params: dict) -> dict:
    """Compute summary statistics for gate/phase parameters."""
    stats = {}
    for key, arr in params.items():
        if arr is None:
            continue
        stats[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "frac_negative": float(np.mean(arr < 0)),
            "frac_lt_0.5": float(np.mean(arr < 0.5)),
            "frac_gt_1.5": float(np.mean(arr > 1.5)),
        }
    return stats


# ── Visualization ───────────────────────────────────────────────────────────

def plot_gate_heatmaps(params_dict: dict, output_dir: Path, prefix: str = ""):
    """Plot heatmaps of gate values across layers and heads."""
    if not HAS_MPL:
        print("[analyze] matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{prefix}AdaptiveRoPE Gate Values", fontsize=14)

    titles = [
        ("encoder_gates_q", "Encoder Gates (Q)"),
        ("encoder_gates_k", "Encoder Gates (K)"),
        ("decoder_gates_q", "Decoder Gates (Q)"),
        ("decoder_gates_k", "Decoder Gates (K)"),
    ]

    for ax, (key, title) in zip(axes.flat, titles):
        arr = params_dict.get(key)
        if arr is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title(title)
            continue
        # Average over frequency dimension: (layers, heads)
        avg = np.mean(arr, axis=-1)
        im = ax.imshow(avg, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=2)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out = output_dir / f"{prefix}gate_heatmaps.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyze] saved {out}")


def plot_phase_distributions(params_dict: dict, output_dir: Path, prefix: str = ""):
    """Plot histograms of phase offset values."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{prefix}AdaptiveRoPE Phase Offset Distributions", fontsize=14)

    titles = [
        ("encoder_phases_q", "Encoder Phases (Q)"),
        ("encoder_phases_k", "Encoder Phases (K)"),
        ("decoder_phases_q", "Decoder Phases (Q)"),
        ("decoder_phases_k", "Decoder Phases (K)"),
    ]

    for ax, (key, title) in zip(axes.flat, titles):
        arr = params_dict.get(key)
        if arr is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_title(title)
            continue
        ax.hist(arr.flatten(), bins=50, color="steelblue", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", label="init (0)")
        ax.set_xlabel("Phase Offset (radians)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    out = output_dir / f"{prefix}phase_distributions.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyze] saved {out}")


def plot_gate_vs_freq(params_dict: dict, output_dir: Path, prefix: str = ""):
    """Plot mean gate value per frequency dimension."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{prefix}Mean Gate Value vs Frequency Index", fontsize=14)

    for ax, (key, title) in zip(axes, [("encoder_gates_q", "Encoder"), ("decoder_gates_q", "Decoder")]):
        arr = params_dict.get(key)
        if arr is None:
            continue
        # arr: (layers, heads, freqs)
        mean_per_freq = np.mean(arr, axis=(0, 1))  # (freqs,)
        std_per_freq = np.std(arr, axis=(0, 1))
        freq_idx = np.arange(len(mean_per_freq))

        ax.plot(freq_idx, mean_per_freq, color="steelblue", linewidth=2)
        ax.fill_between(freq_idx, mean_per_freq - std_per_freq, mean_per_freq + std_per_freq, alpha=0.3)
        ax.axhline(1.0, color="red", linestyle="--", label="init (1.0)")
        ax.set_xlabel("Frequency Index")
        ax.set_ylabel("Mean Gate Value")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(0, 2)

    plt.tight_layout()
    out = output_dir / f"{prefix}gate_vs_freq.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyze] saved {out}")


def plot_crosslingual_gate_comparison(all_params: dict, output_dir: Path):
    """Compare mean gate values across languages."""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    langs = []
    means = []
    stds = []
    for name, params in sorted(all_params.items()):
        # Use encoder gates_q as representative
        arr = params.get("encoder_gates_q")
        if arr is not None:
            langs.append(name.replace("adaptiverope_", "").replace("_s42", "").replace("_s43", "").replace("_s44", ""))
            means.append(np.mean(arr))
            stds.append(np.std(arr))

    x = np.arange(len(langs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=["steelblue", "coral", "seagreen"][:len(langs)])
    ax.axhline(1.0, color="red", linestyle="--", label="RoPE init (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Mean Gate Value")
    ax.set_title("AdaptiveRoPE: Mean Gate Value Across Languages")
    ax.legend()
    ax.set_ylim(0, 2)

    plt.tight_layout()
    out = output_dir / "crosslingual_gate_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyze] saved {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Bootstrap significance tests ─────────────────────────────────────
    print("=" * 60)
    print("1. Bootstrap Significance Tests")
    print("=" * 60)

    sig_results = {}
    for lang_pair, methods in PREDICTIONS.items():
        print(f"\n--- {lang_pair} ---")
        sig_results[lang_pair] = {}

        # Load all predictions
        loaded = {}
        for method, paths in methods.items():
            all_refs, all_hyps = [], []
            for p in paths:
                refs, hyps = _load_predictions(Path(p))
                all_refs.extend(refs)
                all_hyps.extend(hyps)
            loaded[method] = (all_refs, all_hyps)
            bleu = _sentence_bleu(all_refs, all_hyps)
            print(f"  {method}: corpus BLEU = {bleu:.2f} (n={len(all_refs)})")

        # Pairwise comparisons against RoPE
        if "RoPE" in loaded:
            sig_results[lang_pair]["RoPE_baseline"] = {}
            for method, (m_refs, m_hyps) in loaded.items():
                if method == "RoPE":
                    continue
                # For En-De, compare within each seed's predictions, then average p-values
                # For Hi/Bn (single seed), compare directly
                if lang_pair == "en-de":
                    # Load per-seed predictions for RoPE and method
                    rope_paths = methods["RoPE"]
                    method_paths = methods[method]
                    # Only compare seeds that exist for both
                    common = min(len(rope_paths), len(method_paths))
                    p_wins = []
                    diffs = []
                    for i in range(common):
                        r_refs, r_hyps = _load_predictions(Path(rope_paths[i]))
                        m_refs_i, m_hyps_i = _load_predictions(Path(method_paths[i]))
                        # Ensure same length
                        n = min(len(r_refs), len(m_refs_i))
                        result = pairwise_bootstrap_test(r_refs[:n], r_hyps[:n], m_refs_i[:n], m_hyps_i[:n])
                        p_wins.append(result["p_win"])
                        diffs.append(result["mean_diff"])
                    avg_p = np.mean(p_wins)
                    avg_diff = np.mean(diffs)
                    sig_results[lang_pair]["RoPE_baseline"][method] = {
                        "p_win_per_seed": p_wins,
                        "mean_p_win": float(avg_p),
                        "mean_diff": float(avg_diff),
                    }
                    sig = "***" if avg_p > 0.99 else "**" if avg_p > 0.95 else "*" if avg_p > 0.90 else "ns"
                    print(f"  {method} vs RoPE: diff={avg_diff:+.3f}, avg_p(win)={avg_p:.3f} {sig}")
                else:
                    rope_refs, rope_hyps = loaded["RoPE"]
                    n = min(len(rope_refs), len(m_refs), len(m_hyps))
                    result = pairwise_bootstrap_test(rope_refs[:n], rope_hyps[:n], m_refs[:n], m_hyps[:n])
                    sig_results[lang_pair]["RoPE_baseline"][method] = result
                    p = result["p_win"]
                    diff = result["mean_diff"]
                    ci_lo = result["ci_low"]
                    ci_hi = result["ci_high"]
                    sig = "***" if p > 0.99 else "**" if p > 0.95 else "*" if p > 0.90 else "ns"
                    print(f"  {method} vs RoPE: diff={diff:+.3f}, p(win)={p:.3f}, CI=[{ci_lo:+.3f}, {ci_hi:+.3f}] {sig}")

    # Save significance results
    sig_path = out_dir / "bootstrap_significance.json"
    sig_path.write_text(json.dumps(sig_results, indent=2))
    print(f"\n[analyze] saved significance results to {sig_path}")

    # ── 2. AdaptiveRoPE parameter analysis ──────────────────────────────────
    print("\n" + "=" * 60)
    print("2. AdaptiveRoPE Parameter Analysis")
    print("=" * 60)

    all_params = {}
    all_stats = {}
    for name, ckpt_path in CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"  [skip] {name}: checkpoint not found")
            continue
        params = extract_adaptive_rope_params(ckpt_path)
        all_params[name] = params
        stats = param_stats(params)
        all_stats[name] = stats

        print(f"\n  {name}:")
        for key, s in stats.items():
            print(f"    {key}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                  f"median={s['median']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}")

    # Save parameter stats
    stats_path = out_dir / "adaptive_rope_param_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2))
    print(f"\n[analyze] saved parameter stats to {stats_path}")

    # ── 3. Visualizations ───────────────────────────────────────────────────
    if HAS_MPL:
        print("\n" + "=" * 60)
        print("3. Generating Visualizations")
        print("=" * 60)

        for name, params in all_params.items():
            prefix = name.replace("adaptiverope_", "")
            plot_gate_heatmaps(params, out_dir, prefix=f"{prefix}_")
            plot_phase_distributions(params, out_dir, prefix=f"{prefix}_")
            plot_gate_vs_freq(params, out_dir, prefix=f"{prefix}_")

        plot_crosslingual_gate_comparison(all_params, out_dir)
    else:
        print("\n[analyze] matplotlib not installed; skipping visualizations")

    # ── 4. Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. Summary: What the Parameters Tell Us")
    print("=" * 60)

    # Compare across languages
    lang_map = {
        "adaptiverope_de_s42": "En-De (s42)",
        "adaptiverope_de_s43": "En-De (s43)",
        "adaptiverope_de_s44": "En-De (s44)",
        "adaptiverope_hi_s42": "Hi-En (s42)",
        "adaptiverope_bn_s42": "Bn-En (s42)",
    }

    print("\nMean Encoder Gate (Q) per language/seed:")
    for key, label in lang_map.items():
        if key in all_stats:
            s = all_stats[key].get("encoder_gates_q", {})
            print(f"  {label}: {s.get('mean', 'N/A'):.4f} ± {s.get('std', 'N/A'):.4f}")

    print("\nMean Decoder Gate (Q) per language/seed:")
    for key, label in lang_map.items():
        if key in all_stats:
            s = all_stats[key].get("decoder_gates_q", {})
            print(f"  {label}: {s.get('mean', 'N/A'):.4f} ± {s.get('std', 'N/A'):.4f}")

    print("\nPhase offset magnitude (mean |phase|) per language/seed:")
    for key, label in lang_map.items():
        if key in all_params:
            params = all_params[key]
            phases = []
            for pkey in ["encoder_phases_q", "encoder_phases_k", "decoder_phases_q", "decoder_phases_k"]:
                arr = params.get(pkey)
                if arr is not None:
                    phases.append(np.mean(np.abs(arr)))
            if phases:
                print(f"  {label}: {np.mean(phases):.4f}")

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
