"""Phase-Adaptive RoPE (PA-RoPE / AS-RoPE v3).

Extends AS-RoPE v2 with a learnable phase offset per (head, frequency) for
both queries and keys. Standard RoPE always starts rotating from phase 0; here
each head can pick its own starting angle in the complex plane.

  theta_q = pos * base_freqs * gates_q + phase_q
  theta_k = pos * base_freqs * gates_k + phase_k

phase_{q,k} initialized to zero -> identical to AS-RoPE v2 at step 0.
"""

from __future__ import annotations

import torch
from torch import nn


class PARotaryEmbedding(nn.Module):
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
        phase: torch.Tensor,
    ) -> torch.Tensor:
        pos = self.positions[:seq_len].to(device=x.device, dtype=torch.float32)
        base = self.base_freqs.to(device=x.device, dtype=torch.float32)
        g = gates.to(device=x.device, dtype=torch.float32)
        p = phase.to(device=x.device, dtype=torch.float32)

        # theta: (T, H, F) — gate-scaled rotation rate plus per-head phase offset
        theta = (
            pos[:, None, None] * base[None, None, :] * g[None, :, :]
            + p[None, :, :]
        )
        freqs_cis = torch.polar(torch.ones_like(theta), theta).permute(1, 0, 2)
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
        phase_q: torch.Tensor,
        phase_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        return (
            self._apply_rotary(q, seq_len, gates_q, phase_q),
            self._apply_rotary(k, seq_len, gates_k, phase_k),
        )
