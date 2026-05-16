"""Analyze learned parameters of Adaptive RoPE checkpoint.

Loads a trained Adaptive RoPE checkpoint and visualizes the learned per-head
frequency gates and phase offsets, showing what spectral patterns the model
learned compared to standard RoPE initialization.

Usage:
    python -m pipeline.analyze_adaptiverope \
        --checkpoint outputs/checkpoints/asrope3_de/best.pt \
        --out-dir outputs/analysis/adaptiverope_de
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_adaptiverope_params(checkpoint_path: str) -> dict[str, np.ndarray]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    params = {}
    for key in ["gates_q", "gates_k", "phase_q", "phase_k"]:
        # search encoder and decoder attention layers
        for full_key in state:
            if key in full_key:
                params.setdefault(key, []).append(state[full_key].detach().float().numpy())

    # flatten: pick the first found (encoder self-attn) or average across layers
    result = {}
    for key, arrays in params.items():
        result[key] = np.stack(arrays).mean(axis=0)  # (n_heads, n_freqs)
    return result


def plot_heatmap(data: np.ndarray, title: str, cmap: str, vmin: float,
                 vmax: float, out_path: Path) -> None:
    n_heads, n_freqs = data.shape
    fig, ax = plt.subplots(figsize=(max(8, n_freqs // 4), max(4, n_heads // 2)))
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Frequency index (low → high)")
    ax.set_ylabel("Attention head")
    ax.set_title(title)
    ax.set_yticks(range(n_heads))
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_deviation(gates: np.ndarray, label: str, out_path: Path) -> None:
    """Plot how much each head deviates from standard RoPE (gates=1)."""
    deviation = np.abs(gates - 1.0)  # (n_heads, n_freqs)
    head_deviation = deviation.mean(axis=1)  # (n_heads,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(len(head_deviation)), head_deviation)
    axes[0].set_xlabel("Head index")
    axes[0].set_ylabel("Mean |gate - 1|")
    axes[0].set_title(f"{label}: Per-head deviation from RoPE")
    axes[0].set_xticks(range(len(head_deviation)))

    freq_deviation = deviation.mean(axis=0)  # (n_freqs,)
    axes[1].plot(freq_deviation, marker="o", markersize=3)
    axes[1].set_xlabel("Frequency index")
    axes[1].set_ylabel("Mean |gate - 1|")
    axes[1].set_title(f"{label}: Per-frequency deviation from RoPE")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def print_stats(params: dict[str, np.ndarray]) -> None:
    print("\n[analyze] Learned parameter statistics:")
    print(f"  {'param':<12} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    for key, arr in params.items():
        print(f"  {key:<12} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{arr.min():>8.4f} {arr.max():>8.4f}")

    print("\n[analyze] Interpretation:")
    for q_or_k in ["q", "k"]:
        gates = params.get(f"gates_{q_or_k}")
        phase = params.get(f"phase_{q_or_k}")
        if gates is not None:
            max_boost_head = gates.mean(axis=1).argmax()
            max_suppress_head = gates.mean(axis=1).argmin()
            print(f"  gates_{q_or_k}: head {max_boost_head} amplifies most  |  "
                  f"head {max_suppress_head} suppresses most")
        if phase is not None:
            max_shift_head = np.abs(phase).mean(axis=1).argmax()
            print(f"  phase_{q_or_k}: head {max_shift_head} has largest phase shift "
                  f"(mean |phase| = {np.abs(phase[max_shift_head]).mean():.4f} rad)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze Adaptive RoPE learned parameters")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to Adaptive RoPE best.pt checkpoint")
    parser.add_argument("--out-dir", default="outputs/analysis/adaptiverope",
                        help="Directory to save plots")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] loading: {args.checkpoint}")
    params = load_adaptiverope_params(args.checkpoint)

    if not params:
        raise SystemExit("[analyze] No Adaptive RoPE parameters found in checkpoint. "
                         "Make sure this is an adaptiverope checkpoint.")

    print_stats(params)

    # Heatmaps
    for q_or_k in ["q", "k"]:
        gates = params.get(f"gates_{q_or_k}")
        phase = params.get(f"phase_{q_or_k}")

        if gates is not None:
            plot_heatmap(
                gates,
                title=f"Adaptive RoPE — gates_{q_or_k} (learned frequency weights)\n"
                      f"Init=1.0 | >1 = amplify, <1 = suppress",
                cmap="RdBu_r",
                vmin=0.5, vmax=1.5,
                out_path=out_dir / f"gates_{q_or_k}.png",
            )
            plot_deviation(
                gates,
                label=f"gates_{q_or_k}",
                out_path=out_dir / f"gates_{q_or_k}_deviation.png",
            )

        if phase is not None:
            plot_heatmap(
                phase,
                title=f"Adaptive RoPE — phase_{q_or_k} (learned phase offsets, rad)\n"
                      f"Init=0.0 | non-zero = positional shift",
                cmap="PiYG",
                vmin=-0.5, vmax=0.5,
                out_path=out_dir / f"phase_{q_or_k}.png",
            )

    print(f"\n[analyze] All plots saved to: {out_dir}/")
    print("[analyze] Use these to show that Adaptive RoPE learned structured")
    print("          per-head frequency adaptations vs standard RoPE (which would be flat).")


if __name__ == "__main__":
    main()
