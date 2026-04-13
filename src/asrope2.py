"""Adaptive Spectral RoPE v2 (AS-RoPE with decoupled Q/K gates).

Extends AS-RoPE by giving queries and keys independent per-head frequency
gates. In NMT decoder-only models, Q and K play asymmetric roles:
  - Q: "what positional pattern am I looking for?" (target-side attending)
  - K: "what positional signal am I offering?"   (source/target token identity)

Separate gates let each head learn Q-side and K-side frequency profiles
independently, which is especially useful when src (Hindi, free word order)
and tgt (English, strict SVO) are concatenated in the same sequence.

Gate shapes: (n_heads, head_dim // 2) each for Q and K.
Initialized to ones => identical to standard RoPE at step 0.
"""

from __future__ import annotations

import torch
from torch import nn


class ASRotaryEmbeddingV2(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        max_seq_len: int,
        base: float = 10000.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        if d_model != n_heads * head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.n_freqs = head_dim // 2

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer("base_freqs", inv_freq, persistent=False)

    def _apply_rotary(
        self,
        x: torch.Tensor,
        seq_len: int,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding with per-head gates to x of shape (B, H, T, D)."""
        pos = self.positions[:seq_len].to(device=x.device, dtype=torch.float32)
        base = self.base_freqs.to(device=x.device, dtype=torch.float32)
        g = gates.to(device=x.device, dtype=torch.float32)

        # theta: (T, H, n_freqs)
        theta = pos[:, None, None] * base[None, None, :] * g[None, :, :]
        # freqs_cis: (H, T, n_freqs)
        freqs_cis = torch.polar(torch.ones_like(theta), theta).permute(1, 0, 2)
        # unsqueeze batch dim: (1, H, T, n_freqs)
        freqs_cis = freqs_cis.unsqueeze(0)

        x_pair = x.float().contiguous().view(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_pair)
        x_rot = x_complex * freqs_cis
        return torch.view_as_real(x_rot).flatten(start_dim=-2).type_as(x)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        gates_q: torch.Tensor,
        gates_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k: (B, H, T, head_dim)
            gates_q: (H, n_freqs) — per-head frequency gates for queries
            gates_k: (H, n_freqs) — per-head frequency gates for keys
        """
        seq_len = q.size(2)
        return (
            self._apply_rotary(q, seq_len, gates_q),
            self._apply_rotary(k, seq_len, gates_k),
        )
