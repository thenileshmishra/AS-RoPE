"""Decoder-only GPT with pluggable positional encoding.

Supported pe_type values:
  - sinusoidal: fixed sinusoidal added to input embeddings (Vaswani 2017)
  - rope:       standard RoPE (Su 2021)
  - asrope:     AS-RoPE — per-layer learnable Q=K spectral gates
  - asrope2:    AS-RoPE v2 — per-head decoupled Q/K spectral gates
  - asrope3:    PA-RoPE — asrope2 + per-(head, freq) learnable phase offsets
  - ccrope:     Content-Conditioned RoPE — gates derived from causal content summary
  - dsrope:     Dual-Stream RoPE — blend absolute + language-local positions per head
  - none:       no positional encoding

Attention uses F.scaled_dot_product_attention natively (Flash/mem-efficient on CUDA).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.asrope import ASRotaryEmbedding
from src.asrope2 import ASRotaryEmbeddingV2
from src.ccrope import CCRotaryEmbedding
from src.dsrope import DSRotaryEmbedding
from src.parope import PARotaryEmbedding
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
        n_freqs = self.head_dim // 2

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Defaults — most variants leave these None
        self.rope = None
        self.freq_gates = None
        self.freq_gates_q = None
        self.freq_gates_k = None
        self.phase_q = None
        self.phase_k = None
        self.alpha_q = None
        self.alpha_k = None
        self.gate_proj_q = None
        self.gate_proj_k = None

        if pe_type == "rope":
            self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        elif pe_type == "asrope":
            self.rope = ASRotaryEmbedding(d_model, n_heads, self.head_dim, max_seq_len)
            self.freq_gates = nn.Parameter(torch.ones(d_model // 2))
        elif pe_type == "asrope2":
            self.rope = ASRotaryEmbeddingV2(d_model, n_heads, self.head_dim, max_seq_len)
            self.freq_gates_q = nn.Parameter(torch.ones(n_heads, n_freqs))
            self.freq_gates_k = nn.Parameter(torch.ones(n_heads, n_freqs))
        elif pe_type == "asrope3":
            # PA-RoPE: AS-RoPE v2 + learnable phase offsets (zero-init)
            self.rope = PARotaryEmbedding(d_model, n_heads, self.head_dim, max_seq_len)
            self.freq_gates_q = nn.Parameter(torch.ones(n_heads, n_freqs))
            self.freq_gates_k = nn.Parameter(torch.ones(n_heads, n_freqs))
            self.phase_q = nn.Parameter(torch.zeros(n_heads, n_freqs))
            self.phase_k = nn.Parameter(torch.zeros(n_heads, n_freqs))
        elif pe_type == "ccrope":
            # CC-RoPE: content-conditioned gates, projections zero-init -> gates ~ 1.0
            self.rope = CCRotaryEmbedding(d_model, n_heads, self.head_dim, max_seq_len)
            self.gate_proj_q = nn.Linear(d_model, n_heads * n_freqs, bias=False)
            self.gate_proj_k = nn.Linear(d_model, n_heads * n_freqs, bias=False)
            nn.init.zeros_(self.gate_proj_q.weight)
            nn.init.zeros_(self.gate_proj_k.weight)
        elif pe_type == "dsrope":
            # DS-RoPE: dual-stream — alpha blends absolute and language-local positions.
            # alpha=0 logit -> sigmoid=0.5 -> equal blend at init.
            self.rope = DSRotaryEmbedding(d_model, n_heads, self.head_dim, max_seq_len)
            self.alpha_q = nn.Parameter(torch.zeros(n_heads, n_freqs))
            self.alpha_k = nn.Parameter(torch.zeros(n_heads, n_freqs))
            self.freq_gates_q = nn.Parameter(torch.zeros(n_heads, n_freqs))  # softplus(0)~0.69
            self.freq_gates_k = nn.Parameter(torch.zeros(n_heads, n_freqs))

    def forward(
        self,
        x: torch.Tensor,
        pos_local: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pe_type == "rope":
            q, k = self.rope(q, k)
        elif self.pe_type == "asrope":
            q, k = self.rope(q, k, self.freq_gates)
        elif self.pe_type == "asrope2":
            q, k = self.rope(q, k, self.freq_gates_q, self.freq_gates_k)
        elif self.pe_type == "asrope3":
            q, k = self.rope(
                q, k,
                self.freq_gates_q, self.freq_gates_k,
                self.phase_q, self.phase_k,
            )
        elif self.pe_type == "ccrope":
            gates_q, gates_k = CCRotaryEmbedding.compute_gates(
                x, self.gate_proj_q, self.gate_proj_k,
                self.n_heads, self.head_dim // 2,
            )
            q, k = self.rope(q, k, gates_q, gates_k)
        elif self.pe_type == "dsrope":
            if pos_local is None:
                raise RuntimeError("dsrope requires pos_local from GPT.forward")
            pos_abs = torch.arange(
                seq_len, device=x.device, dtype=torch.long
            ).unsqueeze(0).expand(bsz, seq_len)
            q, k = self.rope(
                q, k, pos_abs, pos_local,
                self.alpha_q, self.alpha_k,
                self.freq_gates_q, self.freq_gates_k,
            )

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

    def forward(
        self,
        x: torch.Tensor,
        pos_local: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), pos_local=pos_local)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer with pluggable positional encoding.

    See module docstring for the supported pe_type values.
    """

    _PE_CHOICES = (
        "sinusoidal", "rope", "asrope", "asrope2",
        "asrope3", "ccrope", "dsrope", "none",
    )

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 128,
        mlp_ratio: int = 4,
        pe_type: str = "sinusoidal",
        sep_id: int | None = None,
    ):
        super().__init__()
        if pe_type not in self._PE_CHOICES:
            raise ValueError(
                f"Unknown pe_type='{pe_type}'. Choices: {self._PE_CHOICES}"
            )
        if pe_type == "dsrope" and sep_id is None:
            raise ValueError("pe_type='dsrope' requires sep_id (the [SEP] token id)")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe_type = pe_type
        self.sep_id = sep_id

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

        pos_local = None
        if self.pe_type == "dsrope":
            pos_local = DSRotaryEmbedding.compute_pos_local(input_ids, self.sep_id)

        for block in self.blocks:
            x = block(x, pos_local=pos_local)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss
