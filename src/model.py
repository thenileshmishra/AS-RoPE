"""Decoder-only GPT with pluggable positional encoding.

Currently implemented: sinusoidal, none.
Stubs for: learned, rope, alibi — adding them requires only a new PE module.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.sinusoidal import SinusoidalPositionalEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer with pluggable positional encoding.

    Supported ``pe_type`` values:
      - ``"sinusoidal"`` — fixed sinusoidal (Vaswani et al., 2017).
      - ``"none"``       — no positional encoding.
      - ``"learned"``    — stub (raises ``NotImplementedError``).
      - ``"rope"``       — stub (raises ``NotImplementedError``).
      - ``"alibi"``      — stub (raises ``NotImplementedError``).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 128,
        mlp_ratio: int = 4,
        pe_type: str = "sinusoidal",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe_type = pe_type

        self.token_emb = nn.Embedding(vocab_size, d_model)

        if pe_type == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        elif pe_type == "learned":
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            raise NotImplementedError(
                "pe_type='learned' is not yet implemented. "
                "Add LearnedPE to src/positional_encodings.py and wire it here."
            )
        elif pe_type in ("rope", "alibi"):
            self.pos_emb = None  # PE is handled inside attention layers
            raise NotImplementedError(
                f"pe_type='{pe_type}' is not yet implemented. "
                f"Add the corresponding attention module to src/positional_encodings.py "
                f"and replace self.blocks here."
            )
        elif pe_type == "none":
            self.pos_emb = None
        else:
            raise ValueError(
                f"Unknown pe_type='{pe_type}'. "
                f"Choices: sinusoidal, learned, rope, alibi, none"
            )

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        if self.pos_emb is not None:
            x = self.token_emb(input_ids) + self.pos_emb(seq_len)
        else:
            x = self.token_emb(input_ids)

        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss
