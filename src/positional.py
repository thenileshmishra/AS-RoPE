"""Positional encodings: RoPE, Adaptive RoPE, Sinusoidal.

RoPE uses real sin/cos arithmetic — no torch.polar / view_as_complex,
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
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2).type_as(x)


class RoPE(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Encodes position by rotating Q/K in 2D planes. Each pair (2i, 2i+1)
    is rotated by theta_p = p / (10000^(2i/d)).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.size(-2)
        cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)
        return _apply_rot(q, cos, sin), _apply_rot(k, cos, sin)


class AdaptiveRoPE(nn.Module):
    """Adaptive RoPE: learnable per-head frequency gates and phase offsets.

    Extends RoPE with (n_heads, n_freqs) trainable gates and phases.
    Init: gates=1, phase=0 => identical to standard RoPE at step 0.

    The effective rotation angle becomes:
        theta = pos * base_freq * gate + phase

    This allows each head to independently scale and shift its frequency
    usage, revealing task-specific positional preferences via gradient descent.
    """

    def __init__(self, n_heads: int, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.n_freqs = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("positions", torch.arange(max_seq_len).float(), persistent=False)
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
    """Sinusoidal positional encoding (Vaswani et al., 2017).

    Applied to input embeddings before attention (not to Q/K).
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].unsqueeze(0)


class ALiBi(nn.Module):
    """Attention with Linear Biases (Press et al., 2022).

    Replaces rotary PE with a static linear bias applied to attention scores.
    Bias = -m * |i - j| where m is a head-specific slope.
    No position-dependent rotation of Q/K — instead we add position bias to logits.
    """

    def __init__(self, n_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        # Slopes: geometric sequence from 2^(-8/n_heads) to 2^(-8)
        m = 2.0 ** (-8.0 / n_heads)
        slopes = torch.tensor([m ** (i + 1) for i in range(n_heads)], dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)  # (H,)

        # Precompute distance bias matrix: -slope * distance
        positions = torch.arange(max_seq_len)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        distance = distance.abs().unsqueeze(0)  # (1, T, T)
        # bias will be slopes[:, None, None] * distance  => (H, T, T)
        self.register_buffer("distance", distance, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # ALiBi does not rotate Q/K; bias is added in attention score.
        # We return Q/K unchanged and expect the attention module to add the bias.
        return q, k

    def get_bias(self, seq_len: int, n_heads: int | None = None) -> torch.Tensor:
        """Return ALiBi bias of shape (1, H, T, T) to add to attention scores."""
        T = seq_len
        dist = self.distance[:, :T, :T]  # (1, T, T)
        slopes = self.slopes.view(1, -1, 1, 1)  # (1, H, 1, 1)
        return -slopes * dist  # (1, H, T, T)


class GatesOnlyAdaptiveRoPE(nn.Module):
    """AdaptiveRoPE ablation: learnable gates only, phases frozen at 0."""

    def __init__(self, n_heads: int, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.n_freqs = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("positions", torch.arange(max_seq_len).float(), persistent=False)
        self.register_buffer("base_freqs", inv_freq, persistent=False)
        self.gates_q = nn.Parameter(torch.ones(n_heads, self.n_freqs))
        self.gates_k = nn.Parameter(torch.ones(n_heads, self.n_freqs))
        # Phases are NOT parameters — frozen at 0
        self.register_buffer("phase_q", torch.zeros(n_heads, self.n_freqs), persistent=False)
        self.register_buffer("phase_k", torch.zeros(n_heads, self.n_freqs), persistent=False)

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


class PhasesOnlyAdaptiveRoPE(nn.Module):
    """AdaptiveRoPE ablation: learnable phases only, gates frozen at 1."""

    def __init__(self, n_heads: int, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.n_freqs = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("positions", torch.arange(max_seq_len).float(), persistent=False)
        self.register_buffer("base_freqs", inv_freq, persistent=False)
        # Gates are NOT parameters — frozen at 1
        self.register_buffer("gates_q", torch.ones(n_heads, self.n_freqs), persistent=False)
        self.register_buffer("gates_k", torch.ones(n_heads, self.n_freqs), persistent=False)
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


class NoOpPE(nn.Module):
    """Identity PE for attention layers when sinusoidal is used.

    Sinusoidal PE is applied at embedding level in EncoderDecoder,
    so attention layers pass Q/K through unchanged.
    """
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k


def apply_position_interpolation(model: nn.Module, scale: float) -> None:
    """Apply Position Interpolation (PI) to all RoPE modules in a model.

    PI scales the base frequency by `scale`, effectively compressing the
    position indices so that longer sequences fit within the trained context.
    Chen et al., 2023: https://arxiv.org/abs/2306.15595
    """
    for module in model.modules():
        if isinstance(module, RoPE):
            device = module.cos_cache.device
            head_dim = module.cos_cache.shape[-1] * 2
            new_base = 10000.0 * scale
            inv_freq = 1.0 / (new_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            positions = torch.arange(module.cos_cache.shape[0], dtype=torch.float32)
            freqs = torch.outer(positions, inv_freq)
            module.cos_cache = freqs.cos().to(device)
            module.sin_cache = freqs.sin().to(device)


def build_pe(pe_type: str, n_heads: int, head_dim: int, max_seq_len: int) -> nn.Module:
    """Factory for attention-level positional encodings."""
    if pe_type == "rope":
        return RoPE(head_dim, max_seq_len)
    if pe_type in ("adaptiverope", "asrope3", "asrope2"):  # legacy aliases
        return AdaptiveRoPE(n_heads, head_dim, max_seq_len)
    if pe_type == "sinusoidal":
        return NoOpPE()
    if pe_type == "alibi":
        return ALiBi(n_heads, head_dim, max_seq_len)
    if pe_type == "gatesonly":
        return GatesOnlyAdaptiveRoPE(n_heads, head_dim, max_seq_len)
    if pe_type == "phasesonly":
        return PhasesOnlyAdaptiveRoPE(n_heads, head_dim, max_seq_len)
    raise ValueError(f"unknown pe_type={pe_type!r}; expected rope | adaptiverope | sinusoidal | alibi | gatesonly | phasesonly")
