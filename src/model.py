from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from src.positional_encodings.alibi import ALiBiBias
from src.positional_encodings.as_rope import ASRotaryEmbedding
from src.positional_encodings.ntk_scaled_rope import NTKScaledRotaryEmbedding
from src.positional_encodings.rope import RotaryEmbedding
from src.positional_encodings.scaled_rope import ScaledRotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        positional_encoding: str = "rope",
        use_as_rope: bool = False,
        use_scaled_rope: bool = False,
        allow_negative_gates: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if positional_encoding not in {"rope", "scaled_rope", "as_rope", "alibi", "ntk_scaled_rope"}:
            raise ValueError(
                f"Unsupported positional_encoding='{positional_encoding}'. "
                "Use one of: rope, scaled_rope, as_rope, alibi, ntk_scaled_rope"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.positional_encoding = positional_encoding
        self.use_as_rope = positional_encoding == "as_rope"
        self.use_scaled_rope = positional_encoding == "scaled_rope"
        self.use_alibi = positional_encoding == "alibi"
        self.use_ntk_scaled_rope = positional_encoding == "ntk_scaled_rope"

        if self.use_as_rope:
            self.rope = ASRotaryEmbedding(
                d_model=d_model,
                n_heads=n_heads,
                head_dim=self.head_dim,
                max_seq_len=max_seq_len,
                allow_negative_gates=allow_negative_gates,
            )
        elif self.use_scaled_rope:
            self.rope = ScaledRotaryEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len)
        elif self.use_ntk_scaled_rope:
            self.rope = NTKScaledRotaryEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len)
        elif self.use_alibi:
            self.alibi = ALiBiBias(n_heads=n_heads, max_seq_len=max_seq_len)
        else:
            self.rope = RotaryEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        freq_gates: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, T, C]
        bsz, seq_len, _ = x.shape

        # Project to Q, K, V and split into heads.
        # Each becomes [B, H, T, D]
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply positional encoding to attention.
        if self.use_as_rope:
            if freq_gates is None:
                raise ValueError("freq_gates is required when use_as_rope=True")
            q, k = self.rope(q, k, freq_gates)
        elif self.use_scaled_rope:
            if gamma is None:
                raise ValueError("gamma is required when use_scaled_rope=True")
            q, k = self.rope(q, k, gamma)
        elif not self.use_alibi:
            q, k = self.rope(q, k)

        # Scaled dot-product causal attention.
        # Scores: [B, H, T, T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.use_alibi:
            scores = scores + self.alibi(seq_len=seq_len, device=x.device, dtype=scores.dtype)

        # Causal mask so token i can only attend to <= i.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # [B, H, T, D]

        # Merge heads back: [B, T, C]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int,
        max_seq_len: int,
        positional_encoding: str = "rope",
        use_as_rope: bool = False,
        use_scaled_rope: bool = False,
        allow_negative_gates: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            max_seq_len,
            positional_encoding=positional_encoding,
            use_as_rope=use_as_rope,
            use_scaled_rope=use_scaled_rope,
            allow_negative_gates=allow_negative_gates,
        )
        self.ln2 = nn.LayerNorm(d_model)

        hidden_dim = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_gates: torch.Tensor | None = None,
        gamma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), freq_gates=freq_gates, gamma=gamma)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        max_seq_len: int = 1024,
        mlp_ratio: int = 4,
        positional_encoding: str = "rope",
        use_as_rope: bool = False,
        use_scaled_rope: bool = False,
        as_rope_per_layer_gates: bool = False,
        allow_negative_gates: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        if use_as_rope and use_scaled_rope:
            raise ValueError("use_as_rope and use_scaled_rope cannot both be True")
        if positional_encoding not in {"rope", "scaled_rope", "as_rope", "alibi", "ntk_scaled_rope"}:
            raise ValueError(
                f"Unsupported positional_encoding='{positional_encoding}'. "
                "Use one of: rope, scaled_rope, as_rope, alibi, ntk_scaled_rope"
            )

        if use_as_rope:
            positional_encoding = "as_rope"
        elif use_scaled_rope:
            positional_encoding = "scaled_rope"

        self.positional_encoding = positional_encoding
        self.use_as_rope = positional_encoding == "as_rope"
        self.use_scaled_rope = positional_encoding == "scaled_rope"
        self.use_alibi = positional_encoding == "alibi"
        self.use_ntk_scaled_rope = positional_encoding == "ntk_scaled_rope"
        self.as_rope_per_layer_gates = as_rope_per_layer_gates
        self.allow_negative_gates = allow_negative_gates

        self.freq_gates = None
        if self.use_as_rope:
            if self.as_rope_per_layer_gates:
                self.freq_gates = nn.Parameter(torch.ones(n_layers, d_model // 2))
            else:
                self.freq_gates = nn.Parameter(torch.ones(d_model // 2))

        self.gamma = None
        if self.use_scaled_rope:
            self.gamma = nn.Parameter(torch.tensor(1.0))

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                    positional_encoding=positional_encoding,
                    use_as_rope=use_as_rope,
                    use_scaled_rope=use_scaled_rope,
                    allow_negative_gates=allow_negative_gates,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Optional weight tying (common for GPT models).
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        input_ids: [B, T]
        targets:   [B, T] (optional, for next-token loss)

        Returns:
            logits: [B, T, vocab_size]
            loss: scalar or None
        """
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        # Token embedding.
        x = self.token_emb(input_ids)  # [B, T, C]

        # Decoder stack.
        for layer_idx, block in enumerate(self.blocks):
            layer_freq_gates = self.freq_gates
            if self.use_as_rope and self.as_rope_per_layer_gates and self.freq_gates is not None:
                layer_freq_gates = self.freq_gates[layer_idx]
            x = block(x, freq_gates=layer_freq_gates, gamma=self.gamma)

        # Final normalization and vocabulary projection.
        x = self.final_ln(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )

        return logits, loss
