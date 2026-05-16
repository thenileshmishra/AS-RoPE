"""Generate all figures and metrics needed for the paper.

Produces:
  1. Training loss curves      → outputs/paper/fig_train_loss.png
  2. Frequency spectrum plot   → outputs/paper/fig_freq_spectrum.png
  3. Per-layer gate heatmap    → outputs/paper/fig_gate_layers.png
  4. COMET scores              → outputs/paper/comet_scores.json
  5. Full results table (CSV)  → outputs/paper/results_table.csv

Usage:
    python -m pipeline.paper_figures
"""

from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path("outputs/paper")
OUT.mkdir(parents=True, exist_ok=True)

ROPE_CKPT    = "outputs/checkpoints/rope_de/best.pt"
ADAP_CKPT    = "outputs/checkpoints/asrope3_de/best.pt"
SINU_CKPT    = "outputs/checkpoints/sinusoidal_de/best.pt"

ROPE_LOG     = "outputs/logs/rope_de/metrics.jsonl"
ADAP_LOG     = "outputs/logs/asrope3_de/metrics.jsonl"
SINU_LOG     = "outputs/logs/sinusoidal_de/metrics.jsonl"

ROPE_PREDS   = "outputs/metrics/rope_de_eval/predictions.tsv"
ADAP_PREDS   = "outputs/metrics/asrope3_de_eval/predictions.tsv"
SINU_PREDS   = "outputs/metrics/sinusoidal_de_eval/predictions.tsv"


# ── 1. Training loss curves ────────────────────────────────────────────────

def plot_train_loss():
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = [
        (ROPE_LOG,  "RoPE",          "#1f77b4", "-"),
        (ADAP_LOG,  "Adaptive RoPE", "#d62728", "-"),
        (SINU_LOG,  "Sinusoidal",    "#2ca02c", "--"),
    ]

    for log_path, label, color, ls in configs:
        lines = [json.loads(l) for l in open(log_path)]
        # sample every 500 steps to avoid overplotting
        sampled = [l for l in lines if l["step"] % 500 == 0]
        steps  = [l["step"] for l in sampled]
        losses = [l["train_loss"] for l in sampled]
        ax.plot(steps, losses, label=label, color=color, linestyle=ls, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss Convergence — WMT14 En→De", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=1.0)
    fig.tight_layout()
    out = OUT / "fig_train_loss.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[paper_figures] saved: {out}")


# ── 2. Frequency spectrum plot ─────────────────────────────────────────────

def plot_freq_spectrum():
    ckpt  = torch.load(ADAP_CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # Collect gates_q from all encoder layers
    gates_q = [v.detach().float() for k, v in state.items()
                if "encoder" in k and k.endswith("gates_q")]
    gates_k = [v.detach().float() for k, v in state.items()
                if "encoder" in k and k.endswith("gates_k")]

    if not gates_q:
        print("[paper_figures] No encoder gates found — skipping freq spectrum")
        return

    # mean over layers and heads → (n_freqs,)
    mean_gates_q = torch.stack(gates_q).mean(dim=(0, 1)).numpy()
    mean_gates_k = torch.stack(gates_k).mean(dim=(0, 1)).numpy()

    cfg       = ckpt["config"]
    head_dim  = int(cfg["d_model"]) // int(cfg["n_heads"])
    n_freqs   = head_dim // 2
    base      = 10000.0
    inv_freq  = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))  # (n_freqs,)

    rope_freq     = inv_freq                        # standard RoPE (gate=1)
    adap_freq_q   = inv_freq * mean_gates_q         # Adaptive RoPE query
    adap_freq_k   = inv_freq * mean_gates_k         # Adaptive RoPE key

    fig, ax = plt.subplots(figsize=(9, 5))
    freq_idx = np.arange(n_freqs)

    ax.plot(freq_idx, rope_freq,   "k--",  linewidth=2,   label="RoPE (uniform, gate=1)")
    ax.plot(freq_idx, adap_freq_q, color="#d62728", linewidth=2, label="Adaptive RoPE — Query")
    ax.plot(freq_idx, adap_freq_k, color="#ff7f0e", linewidth=2, label="Adaptive RoPE — Key",  linestyle="--")

    ax.fill_between(freq_idx, rope_freq, adap_freq_q, alpha=0.12, color="#d62728",
                    label="Suppressed region (Q)")

    ax.set_xlabel("Frequency Index  (low ← → high)", fontsize=12)
    ax.set_ylabel("Effective Frequency  (inv_freq × gate)", fontsize=12)
    ax.set_title("Effective RoPE Frequency Spectrum\n"
                 "Adaptive RoPE learns to suppress high-freq components", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_freq_spectrum.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[paper_figures] saved: {out}")


# ── 3. Per-layer gate heatmap ─────────────────────────────────────────────

def plot_gate_layers():
    ckpt  = torch.load(ADAP_CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    # Collect per-layer encoder gates_q, averaged over heads → (n_layers, n_freqs)
    enc_gates = {}
    for k, v in state.items():
        if "encoder" in k and k.endswith("gates_q"):
            # key pattern: encoder.N.attn.pe.gates_q
            parts = k.split(".")
            layer_idx = int(parts[1])
            enc_gates[layer_idx] = v.detach().float().mean(dim=0).numpy()  # mean over heads

    if not enc_gates:
        print("[paper_figures] No per-layer encoder gates — skipping layer plot")
        return

    n_layers = max(enc_gates) + 1
    n_freqs  = next(iter(enc_gates.values())).shape[0]
    matrix   = np.stack([enc_gates[i] for i in range(n_layers)])  # (n_layers, n_freqs)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=0.85, vmax=1.05)
    ax.set_xlabel("Frequency Index  (low ← → high)", fontsize=12)
    ax.set_ylabel("Encoder Layer", fontsize=12)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"Layer {i}" for i in range(n_layers)])
    ax.set_title("Per-Layer Learned Gates (Adaptive RoPE, Encoder)\n"
                 "Values < 1.0 = suppressed, > 1.0 = amplified", fontsize=12)
    plt.colorbar(im, ax=ax, label="Gate value")
    fig.tight_layout()
    out = OUT / "fig_gate_layers.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[paper_figures] saved: {out}")


# ── 4. COMET scores ────────────────────────────────────────────────────────

def compute_comet():
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("[paper_figures] COMET not installed — skipping. Run: pip install unbabel-comet")
        return

    model_path = download_model("Unbabel/wmt22-comet-da")
    model      = load_from_checkpoint(model_path)

    results = {}
    for label, pred_file in [
        ("sinusoidal",   SINU_PREDS),
        ("rope",         ROPE_PREDS),
        ("adaptiverope", ADAP_PREDS),
    ]:
        srcs, refs, hyps = [], [], []
        for line in open(pred_file, encoding="utf-8"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                srcs.append(parts[0])
                refs.append(parts[1])
                hyps.append(parts[2])

        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(srcs, hyps, refs)]
        output = model.predict(data, batch_size=32, gpus=1 if torch.cuda.is_available() else 0)
        score  = float(output.system_score)
        results[label] = round(score, 4)
        print(f"[paper_figures] COMET {label}: {score:.4f}")

    out = OUT / "comet_scores.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[paper_figures] saved: {out}")
    return results


# ── 5. Results table CSV ───────────────────────────────────────────────────

def build_results_table(comet_scores: dict | None = None):
    import csv

    eval_map = {
        "sinusoidal":   "outputs/metrics/sinusoidal_de_eval/eval_summary.json",
        "rope":         "outputs/metrics/rope_de_eval/eval_summary.json",
        "adaptiverope": "outputs/metrics/asrope3_de_eval/eval_summary.json",
    }
    train_map = {
        "sinusoidal":   "outputs/logs/sinusoidal_de/run_summary.json",
        "rope":         "outputs/logs/rope_de/run_summary.json",
        "adaptiverope": "outputs/logs/asrope3_de/run_summary.json",
    }

    rows = []
    for pe in ["sinusoidal", "rope", "adaptiverope"]:
        ev = json.load(open(eval_map[pe]))["overall"]
        tr = json.load(open(train_map[pe]))
        comet = comet_scores.get(pe, "—") if comet_scores else "—"
        rows.append({
            "Model":      pe,
            "Steps":      tr["total_steps"],
            "Val Loss":   round(tr["best_val_loss"], 4),
            "BLEU":       round(ev["bleu"], 2),
            "chrF":       round(ev["chrf"], 2),
            "TER":        round(ev["ter"], 2),
            "COMET":      comet,
            "Time (hr)":  round(tr["total_time_sec"] / 3600, 2),
        })

    out = OUT / "results_table.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[paper_figures] Results table:")
    header = f"{'Model':<14} {'Steps':>6} {'ValLoss':>9} {'BLEU':>6} {'chrF':>6} {'TER':>6} {'COMET':>7} {'Time':>6}"
    print(header); print("-" * len(header))
    for r in rows:
        print(f"{r['Model']:<14} {r['Steps']:>6} {r['Val Loss']:>9} {r['BLEU']:>6} "
              f"{r['chrF']:>6} {r['TER']:>6} {str(r['COMET']):>7} {r['Time (hr)']:>6}")
    print(f"[paper_figures] saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[paper_figures] Generating all paper figures...\n")
    plot_train_loss()
    plot_freq_spectrum()
    plot_gate_layers()
    comet = compute_comet()
    build_results_table(comet)
    print(f"\n[paper_figures] All outputs in: {OUT}/")
