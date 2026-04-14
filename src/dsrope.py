"""Dual-Stream RoPE (DS-RoPE).

Maintains two parallel position streams and lets each (head, frequency) learn
its own blend between them:

  - pos_abs:   standard sequence index (0, 1, 2, ..., T-1)
  - pos_local: language-local index that resets after the [SEP] boundary
               between source (e.g. Hindi) and target (e.g. English)

  alpha = sigmoid(alpha_param)        # (H, F), bounded (0, 1)
  pos_eff = alpha * pos_abs + (1 - alpha) * pos_local
  theta   = pos_eff * base_freqs * softplus(gates)

Heads can specialize: alpha->1 yields global absolute attention (good for
cross-sentence alignment), alpha->0 yields language-local attention (good for
within-language syntactic structure).

Designed for decoder-only NMT where source and target are concatenated in a
single sequence and the standard RoPE position counter conflates them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DSRotaryEmbedding(nn.Module):
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
        self.n_freqs = head_dim // 2
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("base_freqs", inv_freq, persistent=False)

    @staticmethod
    def compute_pos_local(input_ids: torch.Tensor, sep_id: int) -> torch.Tensor:
        """Per-batch language-local positions that reset after each [SEP].

        For each position t, returns t - (index of last sep token at or before t) - 1,
        clamped to >= 0. Tokens before any sep keep their absolute index.
        """
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)
        is_sep = input_ids == sep_id
        sep_positions = torch.where(
            is_sep, pos[None, :].expand(B, T), torch.full_like(input_ids, -1)
        )
        last_sep, _ = torch.cummax(sep_positions, dim=1)  # (B, T)
        pos_local = (pos[None, :] - last_sep - 1).clamp(min=0)
        return pos_local.long()

    def _apply_rotary(
        self,
        x: torch.Tensor,
        pos_eff: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        # x:       (B, H, T, head_dim)
        # pos_eff: (B, T, H, F)
        # gates:   (H, F) — softplus inside
        B, H, T, _ = x.shape
        base = self.base_freqs.to(device=x.device, dtype=torch.float32)
        g = F.softplus(gates).to(device=x.device, dtype=torch.float32)
        p = pos_eff.to(device=x.device, dtype=torch.float32)

        # theta: (B, T, H, F)
        theta = p * base[None, None, None, :] * g[None, None, :, :]
        freqs_cis = torch.polar(torch.ones_like(theta), theta).permute(0, 2, 1, 3)

        x_pair = x.float().contiguous().view(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_pair)
        x_rot = x_complex * freqs_cis
        return torch.view_as_real(x_rot).flatten(start_dim=-2).type_as(x)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_abs: torch.Tensor,
        pos_local: torch.Tensor,
        alpha_q: torch.Tensor,
        alpha_k: torch.Tensor,
        gates_q: torch.Tensor,
        gates_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q, k:       (B, H, T, head_dim)
            pos_abs:    (B, T) absolute positions
            pos_local:  (B, T) language-local positions (resets after [SEP])
            alpha_q/k:  (H, F) unconstrained -> sigmoid for blend in (0, 1)
            gates_q/k:  (H, F) frequency gates (softplus inside)
        """
        aq = torch.sigmoid(alpha_q)
        ak = torch.sigmoid(alpha_k)
        pa = pos_abs[:, :, None, None].float()
        pl = pos_local[:, :, None, None].float()
        # pos_eff: (B, T, H, F)
        pos_eff_q = aq[None, None, :, :] * pa + (1.0 - aq[None, None, :, :]) * pl
        pos_eff_k = ak[None, None, :, :] * pa + (1.0 - ak[None, None, :, :]) * pl
        return (
            self._apply_rotary(q, pos_eff_q, gates_q),
            self._apply_rotary(k, pos_eff_k, gates_k),
        )
