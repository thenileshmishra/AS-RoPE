"""Decoder-only GPT with pluggable positional encoding.

Supported pe_type values: sinusoidal, rope, asrope, none.
Attention uses F.scaled_dot_product_attention natively (Flash/mem-efficient on CUDA).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.asrope import ASRotaryEmbedding
from src.rope import RotaryEmbedding
from src.sinusoidal import SinusoidalPositionalEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        pe_type: str = "sinusoidal",
        max_seq_len: int = 256,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.pe_type = pe_type

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if pe_type == "rope":
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
            self.freq_gates = None
        elif pe_type == "asrope":
            self.rope = ASRotaryEmbedding(d_model, n_heads, self.head_dim, max_seq_len)
            # Per-layer learnable spectral gates, initialized so theta ≈ base RoPE.
            self.freq_gates = nn.Parameter(torch.ones(d_model // 2))
        else:
            self.rope = None
            self.freq_gates = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pe_type == "rope":
            q, k = self.rope(q, k)
        elif self.pe_type == "asrope":
            q, k = self.rope(q, k, self.freq_gates)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        pe_type: str = "sinusoidal",
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, pe_type, max_seq_len)
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

    pe_type:
      - "sinusoidal": fixed sinusoidal added to input embeddings.
      - "rope":       rotary embeddings applied to q, k inside each attention layer.
      - "asrope":     AS-RoPE with per-layer learnable spectral gates.
      - "none":       no positional encoding.
    """

    _PE_CHOICES = ("sinusoidal", "rope", "asrope", "none")

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
        if pe_type not in self._PE_CHOICES:
            raise ValueError(
                f"Unknown pe_type='{pe_type}'. Choices: {self._PE_CHOICES}"
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe_type = pe_type

        self.token_emb = nn.Embedding(vocab_size, d_model)

        if pe_type == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        else:
            self.pos_emb = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, mlp_ratio, pe_type, max_seq_len)
                for _ in range(n_layers)
            ]
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
