"""Positional encodings: RoPE, AsRope2 (per-head Q/K gates),
AsRope3 (per-head Q/K gates + learnable phase offsets)."""

from __future__ import annotations

import torch
from torch import nn


def _build_freqs_cis(head_dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.register_buffer("freqs_cis", _build_freqs_cis(head_dim, max_seq_len, base), persistent=False)

    def _rot(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        freqs_cis = self.freqs_cis[:seq_len].to(x.device)
        x_pair = x.float().contiguous().view(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_pair)
        x_rot = x_complex * freqs_cis.unsqueeze(0).unsqueeze(0)
        return torch.view_as_real(x_rot).flatten(start_dim=-2).type_as(x)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rot(q), self._rot(k)


class AsRope2(nn.Module):
    """Per-head, decoupled Q/K frequency gates. Initialized to ones => identical to RoPE at step 0."""

    def __init__(self, n_heads: int, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.n_freqs = head_dim // 2

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer("base_freqs", inv_freq, persistent=False)

        self.gates_q = nn.Parameter(torch.ones(n_heads, self.n_freqs))
        self.gates_k = nn.Parameter(torch.ones(n_heads, self.n_freqs))

    def _rot(self, x: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(2)
        pos = self.positions[:seq_len].to(device=x.device, dtype=torch.float32)
        base = self.base_freqs.to(device=x.device, dtype=torch.float32)
        g = gates.to(device=x.device, dtype=torch.float32)
        theta = pos[:, None, None] * base[None, None, :] * g[None, :, :]
        freqs_cis = torch.polar(torch.ones_like(theta), theta).permute(1, 0, 2).unsqueeze(0)
        x_pair = x.float().contiguous().view(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_pair)
        return torch.view_as_real(x_complex * freqs_cis).flatten(start_dim=-2).type_as(x)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._rot(q, self.gates_q), self._rot(k, self.gates_k)


class AsRope3(nn.Module):
    """AsRope2 + learnable phase offsets per head per frequency."""

    def __init__(self, n_heads: int, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.n_freqs = head_dim // 2

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        self.register_buffer("positions", positions, persistent=False)
        self.register_buffer("base_freqs", inv_freq, persistent=False)

        self.gates_q = nn.Parameter(torch.ones(n_heads, self.n_freqs))
        self.gates_k = nn.Parameter(torch.ones(n_heads, self.n_freqs))
        self.phase_q = nn.Parameter(torch.zeros(n_heads, self.n_freqs))
        self.phase_k = nn.Parameter(torch.zeros(n_heads, self.n_freqs))

    def _rot(self, x: torch.Tensor, gates: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(2)
        pos = self.positions[:seq_len].to(device=x.device, dtype=torch.float32)
        base = self.base_freqs.to(device=x.device, dtype=torch.float32)
        g = gates.to(device=x.device, dtype=torch.float32)
        p = phase.to(device=x.device, dtype=torch.float32)
        theta = pos[:, None, None] * base[None, None, :] * g[None, :, :] + p[None, :, :]
        freqs_cis = torch.polar(torch.ones_like(theta), theta).permute(1, 0, 2).unsqueeze(0)
        x_pair = x.float().contiguous().view(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_pair)
        return torch.view_as_real(x_complex * freqs_cis).flatten(start_dim=-2).type_as(x)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._rot(q, self.gates_q, self.phase_q),
            self._rot(k, self.gates_k, self.phase_k),
        )


def build_pe(pe_type: str, n_heads: int, head_dim: int, max_seq_len: int) -> nn.Module:
    if pe_type == "rope":
        return RoPE(head_dim, max_seq_len)
    if pe_type == "asrope2":
        return AsRope2(n_heads, head_dim, max_seq_len)
    if pe_type == "asrope3":
        return AsRope3(n_heads, head_dim, max_seq_len)
    raise ValueError(f"unknown pe_type={pe_type!r}; expected one of rope, asrope2, asrope3")
