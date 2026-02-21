import math

import torch
from torch import nn
import torch.nn.functional as F

from as_rope import ASRotaryEmbedding
from rope import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, use_as_rope: bool = False):
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

        self.use_as_rope = use_as_rope
        if self.use_as_rope:
            self.rope = ASRotaryEmbedding(
                d_model=d_model,
                n_heads=n_heads,
                head_dim=self.head_dim,
                max_seq_len=max_seq_len,
            )
        else:
            self.rope = RotaryEmbedding(head_dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor, freq_gates: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, C]
        bsz, seq_len, _ = x.shape

        # Project to Q, K, V and split into heads.
        # Each becomes [B, H, T, D]
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embedding to Q and K.
        if self.use_as_rope:
            if freq_gates is None:
                raise ValueError("freq_gates is required when use_as_rope=True")
            q, k = self.rope(q, k, freq_gates)
        else:
            q, k = self.rope(q, k)

        # Scaled dot-product causal attention.
        # Scores: [B, H, T, T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

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
        use_as_rope: bool = False,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, use_as_rope=use_as_rope)
        self.ln2 = nn.LayerNorm(d_model)

        hidden_dim = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor, freq_gates: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), freq_gates=freq_gates)
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
        use_as_rope: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_as_rope = use_as_rope

        self.freq_gates = None
        if self.use_as_rope:
            self.freq_gates = nn.Parameter(torch.ones(d_model // 2))

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    max_seq_len=max_seq_len,
                    use_as_rope=use_as_rope,
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
        for block in self.blocks:
            x = block(x, freq_gates=self.freq_gates)

        # Final normalization and vocabulary projection.
        x = self.final_ln(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), targets.reshape(-1)
            )

        return logits, loss
