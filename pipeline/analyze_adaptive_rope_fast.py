"""Fast statistical analysis of AdaptiveRoPE parameters and bootstrap significance.

Optimized: precomputes sentence-level BLEU, uses vectorized numpy bootstrap.

Usage:
    python -m pipeline.analyze_adaptive_rope_fast
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Config ──────────────────────────────────────────────────────────────────

CHECKPOINTS = {
    "adaptiverope_de_s42": "outputs/checkpoints/adaptiverope_de_s42/best.pt",
    "adaptiverope_de_s43": "outputs/checkpoints/adaptiverope_de_s43/best.pt",
    "adaptiverope_de_s44": "outputs/checkpoints/adaptiverope_de_s44/best.pt",
    "adaptiverope_hi_s42": "outputs/checkpoints/adaptiverope_hi_s42/best.pt",
    "adaptiverope_bn_s42": "outputs/checkpoints/adaptiverope_bn_s42/best.pt",
}

# For En-De, compare within same seed
PREDICTIONS_EN_DE = {
    "s42": {
        "RoPE": "outputs/metrics/rope_de_eval/predictions.tsv",
        "AdaptiveRoPE": None,  # no s42 for ARoPE on en-de
        "GatesOnly": "outputs/metrics/gatesonly_de_s42_eval/predictions.tsv",
        "PhasesOnly": "outputs/metrics/phasesonly_de_s42_eval/predictions.tsv",
    },
    "s43": {
        "RoPE": "outputs/metrics/rope_de_s43_eval/predictions.tsv",
        "AdaptiveRoPE": "outputs/metrics/adaptiverope_de_s43_eval/predictions.tsv",
        "GatesOnly": "outputs/metrics/gatesonly_de_s43_eval/predictions.tsv",
        "PhasesOnly": "outputs/metrics/phasesonly_de_s43_eval/predictions.tsv",
    },
    "s44": {
        "RoPE": "outputs/metrics/rope_de_s44_eval/predictions.tsv",
        "AdaptiveRoPE": "outputs/metrics/adaptiverope_de_s44_eval/predictions.tsv",
        "GatesOnly": "outputs/metrics/gatesonly_de_s44_eval/predictions.tsv",
        "PhasesOnly": "outputs/metrics/phasesonly_de_s44_eval/predictions.tsv",
    },
}

PREDICTIONS_HI = {
    "RoPE": "outputs/metrics/rope_hi_s42_eval/predictions.tsv",
    "AdaptiveRoPE": "outputs/metrics/adaptiverope_hi_s42_eval/predictions.tsv",
    "GatesOnly": "outputs/metrics/gatesonly_hi_s42_eval/predictions.tsv",
    "PhasesOnly": "outputs/metrics/phasesonly_hi_s42_eval/predictions.tsv",
    "ALiBi": "outputs/metrics/alibi_hi_eval/predictions.tsv",
}

PREDICTIONS_BN = {
    "RoPE": "outputs/metrics/rope_bn_s42_eval/predictions.tsv",
    "AdaptiveRoPE": "outputs/metrics/adaptiverope_bn_s42_eval/predictions.tsv",
    "GatesOnly": "outputs/metrics/gatesonly_bn_s42_eval/predictions.tsv",
    "PhasesOnly": "outputs/metrics/phasesonly_bn_s42_eval/predictions.tsv",
    "ALiBi": "outputs/metrics/alibi_bn_eval/predictions.tsv",
}

N_BOOT = 500
RNG = np.random.default_rng(42)


# ── Fast bootstrap with precomputed sentence BLEU ───────────────────────────

def _load_preds(path: str) -> tuple[list[str], list[str]]:
    refs, hyps = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                refs.append(parts[1])
                hyps.append(parts[2])
    return refs, hyps


def _sentence_bleu_scores(refs: list[str], hyps: list[str]) -> np.ndarray:
    """Compute approximate sentence-level BLEU scores."""
    import sacrebleu
    # Use corpus_bleu on single sentence for consistency with overall metric
    scores = []
    for r, h in zip(refs, hyps):
        try:
            s = sacrebleu.sentence_bleu(h, [r]).score
        except Exception:
            s = 0.0
        scores.append(s)
    return np.array(scores, dtype=np.float64)


def bootstrap_mean_diff(scores_a: np.ndarray, scores_b: np.ndarray, n: int = N_BOOT) -> dict:
    """Vectorized paired bootstrap of mean score difference."""
    m = len(scores_a)
    idx = RNG.integers(0, m, size=(n, m))
    samp_a = scores_a[idx]
    samp_b = scores_b[idx]
    diff = samp_a.mean(axis=1) - samp_b.mean(axis=1)
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "ci_low": float(np.percentile(diff, 2.5)),
        "ci_high": float(np.percentile(diff, 97.5)),
        "p_win": float((diff > 0).mean()),
    }


# ── Parameter extraction ────────────────────────────────────────────────────

def extract_params(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    out = {}
    for k, v in state.items():
        if any(x in k for x in ["gates_q", "gates_k", "phase_q", "phase_k"]):
            out[k] = v.numpy()
    return out


def summarize_params(params: dict) -> dict:
    by_type = {}
    for k, v in params.items():
        key = k.split(".")[-1]  # gates_q, gates_k, phase_q, phase_k
        by_type.setdefault(key, []).append(v)
    stats = {}
    for key, arrs in by_type.items():
        stacked = np.stack(arrs, axis=0)  # (layers, heads, freqs)
        stats[key] = {
            "mean": float(stacked.mean()),
            "std": float(stacked.std()),
            "min": float(stacked.min()),
            "max": float(stacked.max()),
            "median": float(np.median(stacked)),
            "mean_per_layer": [float(x) for x in stacked.mean(axis=(1, 2))],
            "mean_per_head": [float(x) for x in stacked.mean(axis=(0, 2))],
            "mean_per_freq": [float(x) for x in stacked.mean(axis=(0, 1))],
        }
    return stats


# ── Visualization ───────────────────────────────────────────────────────────

def plot_gates(params: dict, out_dir: Path, prefix: str):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    titles = [("gates_q", "Encoder Gates Q"), ("gates_k", "Encoder Gates K"),
              ("decoder", "Decoder Gates Q"), ("decoder", "Decoder Gates K")]
    # Simplified: just plot mean per layer for encoder/decoder gates_q
    enc_gq = [v for k, v in params.items() if "encoder" in k and "gates_q" in k]
    enc_gk = [v for k, v in params.items() if "encoder" in k and "gates_k" in k]
    dec_gq = [v for k, v in params.items() if "decoder" in k and "gates_q" in k]
    dec_gk = [v for k, v in params.items() if "decoder" in k and "gates_k" in k]

    for ax, arr_list, title in zip(axes.flat, [enc_gq, enc_gk, dec_gq, dec_gk],
                                    ["Enc Gates Q", "Enc Gates K", "Dec Gates Q", "Dec Gates K"]):
        if not arr_list:
            continue
        stacked = np.stack(arr_list, axis=0)  # (layers, heads, freqs)
        im = ax.imshow(stacked.mean(axis=-1), aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=2)
        ax.set_title(title)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"{prefix} Gate Heatmaps")
    plt.tight_layout()
    out = out_dir / f"{prefix}_gates.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


def plot_phases(params: dict, out_dir: Path, prefix: str):
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    enc_pq = [v for k, v in params.items() if "encoder" in k and "phase_q" in k]
    enc_pk = [v for k, v in params.items() if "encoder" in k and "phase_k" in k]
    dec_pq = [v for k, v in params.items() if "decoder" in k and "phase_q" in k]
    dec_pk = [v for k, v in params.items() if "decoder" in k and "phase_k" in k]

    for ax, arr_list, title in zip(axes.flat, [enc_pq, enc_pk, dec_pq, dec_pk],
                                    ["Enc Phase Q", "Enc Phase K", "Dec Phase Q", "Dec Phase K"]):
        if not arr_list:
            continue
        stacked = np.stack(arr_list, axis=0)
        ax.hist(stacked.flatten(), bins=60, color="steelblue", edgecolor="white")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Phase offset (rad)")
    fig.suptitle(f"{prefix} Phase Distributions")
    plt.tight_layout()
    out = out_dir / f"{prefix}_phases.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


def plot_crosslingual(all_stats: dict, out_dir: Path):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    gate_means = []
    gate_stds = []
    phase_mags = []
    for name, stats in sorted(all_stats.items()):
        label = name.replace("adaptiverope_", "").replace("_s42", "").replace("_s43", "").replace("_s44", "")
        labels.append(label)
        gq = stats.get("gates_q", {})
        gate_means.append(gq.get("mean", 0))
        gate_stds.append(gq.get("std", 0))
        # Phase magnitude
        pq = stats.get("phase_q", {})
        pk = stats.get("phase_k", {})
        pm = 0
        if pq and pk:
            pm = (abs(pq.get("mean", 0)) + abs(pk.get("mean", 0))) / 2
        phase_mags.append(pm)

    x = np.arange(len(labels))
    ax.bar(x - 0.2, gate_means, 0.4, yerr=gate_stds, capsize=3, label="Gate mean", color="steelblue")
    ax.bar(x + 0.2, phase_mags, 0.4, label="|Phase| mean", color="coral")
    ax.axhline(1.0, color="green", linestyle="--", label="Gate init=1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("AdaptiveRoPE: Learned Parameters Across Languages")
    ax.legend()
    plt.tight_layout()
    out = out_dir / "crosslingual_params.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  saved {out}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AdaptiveRoPE Statistical Analysis (Fast)")
    print("=" * 60)

    # ── 1. Bootstrap significance ───────────────────────────────────────────
    print("\n--- 1. Bootstrap Significance Tests ---")
    sig_all = {}

    # En-De: per-seed comparison
    print("\nEn-De (per seed):")
    sig_all["en-de"] = {}
    for seed, methods in PREDICTIONS_EN_DE.items():
        rope_path = methods.get("RoPE")
        if not rope_path or not Path(rope_path).exists():
            continue
        rope_refs, rope_hyps = _load_preds(rope_path)
        rope_scores = _sentence_bleu_scores(rope_refs, rope_hyps)
        print(f"  Seed {seed}: RoPE n={len(rope_scores)}")
        sig_all["en-de"][seed] = {}
        for method, path in methods.items():
            if method == "RoPE" or not path or not Path(path).exists():
                continue
            m_refs, m_hyps = _load_preds(path)
            m_scores = _sentence_bleu_scores(m_refs, m_hyps)
            n = min(len(rope_scores), len(m_scores))
            result = bootstrap_mean_diff(m_scores[:n], rope_scores[:n])
            sig_all["en-de"][seed][method] = result
            p = result["p_win"]
            diff = result["mean_diff"]
            sig = "***" if p > 0.99 else "**" if p > 0.95 else "*" if p > 0.90 else "ns"
            print(f"    {method} vs RoPE: diff={diff:+.3f}, p(win)={p:.3f} {sig}")

    # Hi-En
    print("\nHi-En:")
    sig_all["hi-en"] = {}
    r_refs, r_hyps = _load_preds(PREDICTIONS_HI["RoPE"])
    r_scores = _sentence_bleu_scores(r_refs, r_hyps)
    print(f"  RoPE n={len(r_scores)}")
    for method, path in PREDICTIONS_HI.items():
        if method == "RoPE":
            continue
        m_refs, m_hyps = _load_preds(path)
        m_scores = _sentence_bleu_scores(m_refs, m_hyps)
        n = min(len(r_scores), len(m_scores))
        result = bootstrap_mean_diff(m_scores[:n], r_scores[:n])
        sig_all["hi-en"][method] = result
        p = result["p_win"]
        diff = result["mean_diff"]
        sig = "***" if p > 0.99 else "**" if p > 0.95 else "*" if p > 0.90 else "ns"
        print(f"  {method} vs RoPE: diff={diff:+.3f}, p(win)={p:.3f} {sig}")

    # Bn-En
    print("\nBn-En:")
    sig_all["bn-en"] = {}
    r_refs, r_hyps = _load_preds(PREDICTIONS_BN["RoPE"])
    r_scores = _sentence_bleu_scores(r_refs, r_hyps)
    print(f"  RoPE n={len(r_scores)}")
    for method, path in PREDICTIONS_BN.items():
        if method == "RoPE":
            continue
        m_refs, m_hyps = _load_preds(path)
        m_scores = _sentence_bleu_scores(m_refs, m_hyps)
        n = min(len(r_scores), len(m_scores))
        result = bootstrap_mean_diff(m_scores[:n], r_scores[:n])
        sig_all["bn-en"][method] = result
        p = result["p_win"]
        diff = result["mean_diff"]
        sig = "***" if p > 0.99 else "**" if p > 0.95 else "*" if p > 0.90 else "ns"
        print(f"  {method} vs RoPE: diff={diff:+.3f}, p(win)={p:.3f} {sig}")

    sig_path = out_dir / "bootstrap_significance.json"
    sig_path.write_text(json.dumps(sig_all, indent=2))
    print(f"\n  Saved significance results to {sig_path}")

    # ── 2. Parameter extraction ─────────────────────────────────────────────
    print("\n--- 2. AdaptiveRoPE Parameter Statistics ---")
    all_params = {}
    all_stats = {}
    for name, ckpt_path in CHECKPOINTS.items():
        if not Path(ckpt_path).exists():
            print(f"  [skip] {name}: not found")
            continue
        t0 = time.time()
        params = extract_params(ckpt_path)
        stats = summarize_params(params)
        all_params[name] = params
        all_stats[name] = stats
        print(f"  {name}:")
        for key, s in stats.items():
            print(f"    {key}: mean={s['mean']:.4f}, std={s['std']:.4f}, median={s['median']:.4f}")
        print(f"    (loaded in {time.time()-t0:.2f}s)")

    stats_path = out_dir / "param_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2))
    print(f"\n  Saved parameter stats to {stats_path}")

    # ── 3. Plots ────────────────────────────────────────────────────────────
    if HAS_MPL:
        print("\n--- 3. Generating Visualizations ---")
        for name, params in all_params.items():
            prefix = name.replace("adaptiverope_", "")
            plot_gates(params, out_dir, prefix)
            plot_phases(params, out_dir, prefix)
        plot_crosslingual(all_stats, out_dir)
    else:
        print("\n  [skip] matplotlib not available")

    # ── 4. Summary interpretation ───────────────────────────────────────────
    print("\n--- 4. Interpretation Summary ---")
    print("\nGate means (<1 = suppression, >1 = amplification):")
    for name, stats in sorted(all_stats.items()):
        label = name.replace("adaptiverope_", "")
        gq = stats.get("gates_q", {})
        gk = stats.get("gates_k", {})
        print(f"  {label}: Q={gq.get('mean', 0):.4f}, K={gk.get('mean', 0):.4f}")

    print("\nPhase magnitudes (deviation from init=0):")
    for name, stats in sorted(all_stats.items()):
        label = name.replace("adaptiverope_", "")
        pq = stats.get("phase_q", {})
        pk = stats.get("phase_k", {})
        pm = 0
        if pq and pk:
            pm = (abs(pq.get("mean", 0)) + abs(pk.get("mean", 0))) / 2
        print(f"  {label}: |phase|={pm:.4f}")

    print("\n" + "=" * 60)
    print("Done. All outputs in outputs/analysis/")
    print("=" * 60)


if __name__ == "__main__":
    main()
