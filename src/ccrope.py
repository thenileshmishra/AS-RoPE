"""Content-Conditioned RoPE (CC-RoPE).

Makes the rotary frequency itself content-aware. Standard RoPE / AS-RoPE
compute theta = f(position) only — the rotation is blind to *what* the token
actually is. CC-RoPE derives per-(batch, position, head, frequency) gates from
a causal cumulative summary of the input embeddings, then multiplies them into
the base rotary phase:

    context[b, t]  = mean(x[b, 0:t+1])           # causal — no lookahead
    gates_q(t)     = 1 + 0.5 * tanh(W_q @ context[b, t])
    gates_k(t)     = 1 + 0.5 * tanh(W_k @ context[b, t])
    theta_{q,k}    = pos * base_freqs * gates_{q,k}(t)

W_{q,k} are zero-initialized -> gates start at 1.0 -> identical to standard
RoPE at step 0. Bounded gate range (0.5, 1.5) avoids degenerate frequencies.
"""

from __future__ import annotations

import torch
from torch import nn


class CCRotaryEmbedding(nn.Module):
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

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_freqs = head_dim // 2
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer("base_freqs", inv_freq, persistent=False)

    @staticmethod
    def compute_gates(
        x: torch.Tensor,
        gate_proj_q: nn.Linear,
        gate_proj_k: nn.Linear,
        n_heads: int,
        n_freqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (gates_q, gates_k) of shape (B, T, H, F) from input x (B, T, d)."""
        B, T, _ = x.shape
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1)
        # Causal cumulative mean — each position sees only its own past
        context = x.cumsum(dim=1) / counts
        gq = gate_proj_q(context).view(B, T, n_heads, n_freqs)
        gk = gate_proj_k(context).view(B, T, n_heads, n_freqs)
        gates_q = 1.0 + 0.5 * torch.tanh(gq)
        gates_k = 1.0 + 0.5 * torch.tanh(gk)
        return gates_q, gates_k

    def _apply_rotary(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        # x:     (B, H, T, head_dim)
        # gates: (B, T, H, F)
        B, H, T, _ = x.shape
        pos = self.positions[:T].to(device=x.device, dtype=torch.float32)
        base = self.base_freqs.to(device=x.device, dtype=torch.float32)
        g = gates.to(device=x.device, dtype=torch.float32)

        # theta: (B, T, H, F)
        theta = pos[None, :, None, None] * base[None, None, None, :] * g
        freqs_cis = torch.polar(torch.ones_like(theta), theta)
        # -> (B, H, T, F) to align with x's head-major layout
        freqs_cis = freqs_cis.permute(0, 2, 1, 3)

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
        return self._apply_rotary(q, gates_q), self._apply_rotary(k, gates_k)
