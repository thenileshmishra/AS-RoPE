"""Deterministic sinusoidal positional embeddings (Vaswani et al., 2017).

Produces a fixed (non-learnable) positional encoding table of shape
[max_seq_len, d_model] that is added to token embeddings before the
transformer blocks.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    The table is registered as a non-persistent buffer so it moves with the
    model device but is not saved as a learnable parameter.
    """

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal PE, got {d_model}")

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [max_seq_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return positional embeddings for positions 0..seq_len-1.

        Returns: [seq_len, d_model]
        """
        return self.pe[:seq_len]
