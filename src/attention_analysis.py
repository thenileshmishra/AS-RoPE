"""Attention entropy and sink-token analysis for encoder-decoder checkpoints.

Loads a trained checkpoint, runs inference on validation data, and extracts
attention weights from decoder self-attention to compute:
  1. Per-head attention entropy
  2. Attention sink detection (first token receiving disproportionate attention)
  3. Entropy degradation vs sequence length

Usage:
    python -m src.attention_analysis \
        --checkpoint outputs/checkpoints/rope_de/best.pt \
        --eval-tsv raw_data/wmt14/val.tsv \
        --out-dir outputs/analysis/attention/rope_de
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.eval import load_model_from_checkpoint
from src.model import EncoderDecoder
from src.mt_data import MTPairDatasetCached, build_collator
from src.tokenizer_utils import build_mt_tokenizer


@torch.no_grad()
def extract_attention_maps(
    model: EncoderDecoder,
    src: torch.Tensor,
    tgt_in: torch.Tensor,
) -> dict[str, list[torch.Tensor]]:
    """Run a forward pass and capture attention weights from decoder self-attention.

    Returns a dict mapping layer index -> list of attention tensors (B, H, T, T).
    """
    # We need to modify the model to return attention weights.
    # Since the model uses F.scaled_dot_product_attention which doesn't return weights,
    # we temporarily replace the attention forward to use manual attention.
    # Instead, let's use forward hooks to capture Q, K, V and compute attention manually.
    pass


class AttentionCapture:
    """Context manager that hooks into decoder self-attention layers to capture Q, K."""

    def __init__(self, model: EncoderDecoder):
        self.model = model
        self.hooks: list = []
        self.qk_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _make_hook(self, layer_idx: int):
        def hook(module, args, output):
            # module is MultiHeadSelfAttention
            # args[0] is input x
            x = args[0]
            B, T, D = x.shape
            qkv = module.qkv(x).view(B, T, 3, module.n_heads, module.head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]
            q, k = module.pe(q, k)
            self.qk_outputs[layer_idx] = (q.detach().cpu(), k.detach().cpu())
        return hook

    def __enter__(self):
        for idx, layer in enumerate(self.model.decoder):
            h = layer.self_attn.register_forward_hook(self._make_hook(idx))
            self.hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def compute_attention_weights(self) -> dict[int, torch.Tensor]:
        """Compute softmax(QK^T / sqrt(d)) for each captured layer."""
        attn_weights = {}
        for layer_idx, (q, k) in self.qk_outputs.items():
            # q, k: (B, H, T, Dh)
            scale = q.size(-1) ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
            # Apply causal mask
            T = scores.size(-1)
            causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
            attn_weights[layer_idx] = weights
        return attn_weights


def compute_entropy(attn_weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Compute attention entropy: H = -sum(p * log(p)) for each query position.

    attn_weights: (B, H, T_q, T_k)
    Returns: (B, H, T_q) entropy values.
    """
    log_p = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_p).sum(dim=-1)  # (B, H, T_q)
    return entropy


def compute_sink_ratio(attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute ratio of attention mass on the first token (sink).

    attn_weights: (B, H, T_q, T_k)
    Returns: (B, H, T_q) ratio of attention on first position.
    """
    first_token_attn = attn_weights[..., 0]  # (B, H, T_q)
    return first_token_attn


def analyze_checkpoint(
    checkpoint_path: str,
    tokenized_val: str,
    out_dir: str,
    device: str,
    max_batches: int = 20,
    batch_size: int = 16,
) -> dict:
    """Load checkpoint, run inference, extract and analyze attention."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model_from_checkpoint(checkpoint_path, device)
    model.eval()

    pad_id = int(cfg["pad_id"])
    bos_id = int(cfg["bos_id"])
    eos_id = int(cfg["eos_id"])

    val_ds = MTPairDatasetCached(tokenized_val)
    collate = build_collator(pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    all_entropies: list[np.ndarray] = []
    all_sink_ratios: list[np.ndarray] = []
    all_seq_lens: list[np.ndarray] = []

    n_layers = len(model.decoder)

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        labels = batch["labels"].to(device)

        # Compute valid lengths (exclude padding)
        valid = (labels != -100).sum(dim=-1)  # (B,)
        seq_lens = valid.cpu().numpy()

        with AttentionCapture(model) as cap:
            _ = model(src, tgt_in)
            attn_weights = cap.compute_attention_weights()

        # Per-layer entropy and sink
        batch_entropies = []
        batch_sinks = []
        for layer_idx in range(n_layers):
            w = attn_weights[layer_idx]  # (B, H, T, T)
            ent = compute_entropy(w).numpy()  # (B, H, T)
            sink = compute_sink_ratio(w).numpy()  # (B, H, T)
            batch_entropies.append(ent)
            batch_sinks.append(sink)

        # Stack layers: (B, L, H, T)
        stacked_ent = np.stack(batch_entropies, axis=1)
        stacked_sink = np.stack(batch_sinks, axis=1)

        # Split into individual samples (since T varies across batches)
        for b in range(stacked_ent.shape[0]):
            all_entropies.append(stacked_ent[b])   # (L, H, T)
            all_sink_ratios.append(stacked_sink[b]) # (L, H, T)
        all_seq_lens.append(seq_lens)

    # Keep as list of individual samples (seq len varies)
    all_entropies_flat = all_entropies   # list of (L, H, T)
    all_sinks_flat = all_sink_ratios     # list of (L, H, T)
    seq_lens = np.concatenate(all_seq_lens)  # (N,)
    n_samples = len(all_entropies_flat)

    n_layers = all_entropies_flat[0].shape[0]
    n_heads = all_entropies_flat[0].shape[1]

    # Summary statistics
    results = {
        "checkpoint": checkpoint_path,
        "pe_type": cfg.get("pe_type"),
        "n_samples": n_samples,
        "n_layers": n_layers,
        "n_heads": n_heads,
    }

    # Per-layer mean entropy (averaged over heads and positions)
    per_layer_entropy = []
    for l in range(n_layers):
        # Collect all valid positions across samples
        ent_vals = []
        sink_vals = []
        for i in range(n_samples):
            t = int(seq_lens[i])
            ent_vals.append(all_entropies_flat[i][l, :, :t].flatten())
            sink_vals.append(all_sinks_flat[i][l, :, :t].flatten())
        layer_ent = np.concatenate(ent_vals)
        layer_sink = np.concatenate(sink_vals)
        per_layer_entropy.append({
            "layer": l,
            "mean_entropy": float(layer_ent.mean()),
            "std_entropy": float(layer_ent.std()),
            "mean_sink_ratio": float(layer_sink.mean()),
        })
    results["per_layer"] = per_layer_entropy

    # Plot entropy per layer
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = [r["layer"] for r in per_layer_entropy]
    means = [r["mean_entropy"] for r in per_layer_entropy]
    ax.plot(layers, means, "o-", linewidth=2, markersize=6)
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Mean Attention Entropy")
    ax.set_title(f"Attention Entropy per Layer — {cfg.get('pe_type', 'unknown')}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "entropy_per_layer.png", dpi=150)
    plt.close(fig)

    # Plot sink ratio per layer
    fig, ax = plt.subplots(figsize=(10, 5))
    sinks = [r["mean_sink_ratio"] for r in per_layer_entropy]
    ax.plot(layers, sinks, "s-", linewidth=2, markersize=6, color="orange")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Mean Attention on First Token (sink ratio)")
    ax.set_title(f"Attention Sink per Layer — {cfg.get('pe_type', 'unknown')}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path / "sink_per_layer.png", dpi=150)
    plt.close(fig)

    # Save JSON
    (out_path / "attention_analysis.json").write_text(json.dumps(results, indent=2))
    print(f"[attention] Saved analysis to {out_path}")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Attention entropy analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenized-val", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    analyze_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenized_val=args.tokenized_val,
        out_dir=args.out_dir,
        device=args.device,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
