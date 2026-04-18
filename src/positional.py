"""Positional encodings: RoPE, AsRope2 (per-head Q/K gates),
AsRope3 (per-head Q/K gates + learnable phase offsets).

All implementations use real sin/cos arithmetic only — no torch.polar /
view_as_complex, so torch.compile can fully fuse every kernel.

Rotation formula (equivalent to complex multiply):
    x_rot[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    x_rot[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
"""

from __future__ import annotations

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
    """Standard RoPE with precomputed cos/sin cache."""

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


class AsRope2(nn.Module):
    """Per-head decoupled Q/K frequency gates. Initialized to ones => identical to RoPE at step 0."""

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

    def _cos_sin(self, seq_len: int, gates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # pos: (T,)  base: (F,)  gates: (H, F)
        # theta: (1, H, T, F)
        pos = self.positions[:seq_len]
        theta = (pos[:, None, None] * self.base_freqs[None, None, :] * gates[None, :, :]).permute(1, 0, 2).unsqueeze(0)
        return theta.cos(), theta.sin()

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.size(-2)
        cos_q, sin_q = self._cos_sin(T, self.gates_q)
        cos_k, sin_k = self._cos_sin(T, self.gates_k)
        return _apply_rot(q, cos_q, sin_q), _apply_rot(k, cos_k, sin_k)


class AsRope3(nn.Module):
    """AsRope2 + learnable phase offsets per head per frequency."""

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


def build_pe(pe_type: str, n_heads: int, head_dim: int, max_seq_len: int) -> nn.Module:
    if pe_type == "rope":
        return RoPE(head_dim, max_seq_len)
    if pe_type == "asrope2":
        return AsRope2(n_heads, head_dim, max_seq_len)
    if pe_type == "asrope3":
        return AsRope3(n_heads, head_dim, max_seq_len)
    raise ValueError(f"unknown pe_type={pe_type!r}; expected one of rope, asrope2, asrope3")
