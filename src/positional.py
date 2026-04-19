"""Positional encodings: RoPE, Adaptive RoPE (learnable variant), and Sinusoidal.

RoPE and RoPE-v3 use real sin/cos arithmetic only — no torch.polar / view_as_complex,
so torch.compile can fully fuse every kernel.

Rotation formula (equivalent to complex multiply):
    x_rot[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    x_rot[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _apply_rot(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary rotation using real arithmetic.

    x   : (B, H, T, D)
    cos : broadcastable to (B, H, T, D//2)
    sin : broadcastable to (B, H, T, D//2)
    """
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) with precomputed cos/sin cache.

    Encodes position by rotating Q/K vectors in 2D planes. Each pair of
    dimensions (2i, 2i+1) is rotated by an angle theta_p = p / (10000^(2i/d))
    where p is the position and d is the head dimension.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)          # (T, D//2)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.size(-2)
        cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)  # (1,1,T,D//2)
        sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)
        return _apply_rot(q, cos, sin), _apply_rot(k, cos, sin)


class AdaptiveRoPE(nn.Module):
    """Adaptive RoPE: Rotary Position Embedding with learnable phase offsets.

    Novel extension of RoPE that adds learnable phase offsets per head per frequency.
    This allows the model to adapt the rotational structure of positional encoding
    during training, potentially capturing task-specific position patterns.

    Parameters:
        gates_q, gates_k: (n_heads, n_freqs) learnable per-head frequency gates (init ones)
        phase_q, phase_k: (n_heads, n_freqs) learnable phase offsets (init zeros)

    Initialization: gates=1.0 and phase=0.0 ⟹ identical to standard RoPE at step 0.
    During training, gradients adjust gates and phases to optimize attention patterns.
    """

    def __init__(self, n_heads: int, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.n_freqs = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer("base_freqs", inv_freq, persistent=False)

        self.gates_q = nn.Parameter(torch.ones(n_heads, self.n_freqs))
        self.gates_k = nn.Parameter(torch.ones(n_heads, self.n_freqs))
        self.phase_q = nn.Parameter(torch.zeros(n_heads, self.n_freqs))
        self.phase_k = nn.Parameter(torch.zeros(n_heads, self.n_freqs))

    def _cos_sin(self, seq_len: int, gates: torch.Tensor,
                 phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.positions[:seq_len]
        theta = (pos[:, None, None] * self.base_freqs[None, None, :] * gates[None, :, :]
                 + phase[None, :, :]).permute(1, 0, 2).unsqueeze(0)
        return theta.cos(), theta.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.size(-2)
        cos_q, sin_q = self._cos_sin(T, self.gates_q, self.phase_q)
        cos_k, sin_k = self._cos_sin(T, self.gates_k, self.phase_k)
        return _apply_rot(q, cos_q, sin_q), _apply_rot(k, cos_k, sin_k)


class Sinusoidal(nn.Module):
    """Standard sinusoidal positional encoding (absolute positions).

    Uses the formula from Vaswani et al. (2017):
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Applied directly to embeddings before attention.
    Note: Does not modify Q/K, applied to input embeddings.
    """

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")

        pe = torch.zeros(max_seq_len, d_model)
        positions = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(positions * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(positions * div_term)
        else:
            pe[:, 1::2] = torch.cos(positions * div_term[:-1])

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal PE to input embeddings."""
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


def build_pe(pe_type: str, n_heads: int, head_dim: int, max_seq_len: int,
             d_model: int = None) -> nn.Module:
    """Factory for positional encodings.

    Args:
        pe_type: "rope", "adaptiverope", or "sinusoidal"
        n_heads: number of attention heads
        head_dim: dimension per head
        max_seq_len: maximum sequence length
        d_model: model dimension (for Sinusoidal)
    """
    if pe_type == "rope":
        return RoPE(head_dim, max_seq_len)
    if pe_type == "adaptiverope":
        return AdaptiveRoPE(n_heads, head_dim, max_seq_len)
    if pe_type == "sinusoidal":
        if d_model is None:
            d_model = n_heads * head_dim
        return Sinusoidal(d_model, max_seq_len)
    raise ValueError(f"unknown pe_type={pe_type!r}; expected 'rope', 'adaptiverope', or 'sinusoidal'")
