"""Encoder-decoder Transformer for MT with Rotary Position Embedding (RoPE).

Rotary is applied on Q/K in self-attention (encoder + decoder). Cross-attention
uses no rotary — target positions are already captured in decoder self-attn Q.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.positional import build_pe


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, pe_type: str, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.pe = build_pe(pe_type, n_heads, self.head_dim, max_seq_len)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
                causal: bool = False) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Dh)
        q, k = self.pe(q, k)

        attn_mask = None
        is_causal = causal
        if key_padding_mask is not None:
            # Build a float mask combining padding (and causal if requested),
            # because SDPA rejects attn_mask + is_causal together.
            pad_mask = key_padding_mask.view(B, 1, 1, T).expand(B, self.n_heads, T, T)
            attn_mask = torch.zeros(B, self.n_heads, T, T, dtype=q.dtype, device=q.device)
            attn_mask = attn_mask.masked_fill(pad_mask, float("-inf"))
            if causal:
                causal_mask = torch.triu(
                    torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=1
                )
                attn_mask = attn_mask.masked_fill(causal_mask.view(1, 1, T, T), float("-inf"))
                is_causal = False

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, enc: torch.Tensor,
                enc_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        S = enc.size(1)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(enc).view(B, S, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn_mask = None
        if enc_key_padding_mask is not None:
            attn_mask = enc_key_padding_mask.view(B, 1, 1, S).expand(B, self.n_heads, T, S)
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(attn_mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int,
                 pe_type: str, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, max_seq_len, pe_type, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_kpm: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), key_padding_mask=src_kpm, causal=False))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int,
                 pe_type: str, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, max_seq_len, pe_type, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc: torch.Tensor,
                tgt_kpm: torch.Tensor | None, src_kpm: torch.Tensor | None) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.ln1(x), key_padding_mask=tgt_kpm, causal=True))
        x = x + self.drop(self.cross_attn(self.ln2(x), enc, enc_key_padding_mask=src_kpm))
        x = x + self.drop(self.ff(self.ln3(x)))
        return x


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 384,
        n_heads: int = 6,
        d_ff: int = 1536,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        max_seq_len: int = 256,
        pe_type: str = "rope",
        dropout: float = 0.1,
        tie_embeddings: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, max_seq_len, pe_type, dropout)
            for _ in range(n_enc_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, max_seq_len, pe_type, dropout)
            for _ in range(n_dec_layers)
        ])
        self.enc_ln = nn.LayerNorm(d_model)
        self.dec_ln = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.use_checkpoint = use_checkpoint
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _pad_mask(self, ids: torch.Tensor) -> torch.Tensor:
        return ids == self.pad_id  # True at PAD

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_kpm = self._pad_mask(src)
        x = self.emb_drop(self.embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, src_kpm, use_reentrant=False)
            else:
                x = layer(x, src_kpm)
        return self.enc_ln(x), src_kpm

    def decode(self, tgt: torch.Tensor, enc: torch.Tensor, src_kpm: torch.Tensor) -> torch.Tensor:
        tgt_kpm = self._pad_mask(tgt)
        x = self.emb_drop(self.embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, enc, tgt_kpm, src_kpm, use_reentrant=False)
            else:
                x = layer(x, enc, tgt_kpm, src_kpm)
        return self.dec_ln(x)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        enc, src_kpm = self.encode(src)
        h = self.decode(tgt_in, enc, src_kpm)
        return self.lm_head(h)

    @torch.no_grad()
    def generate_beam(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        beam_size: int = 5,
        max_new_tokens: int = 128,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        """Beam search decoding. Processes one sentence at a time; returns (B, T) best hypotheses."""
        self.eval()
        device = src.device
        results: list[list[int]] = []

        for b in range(src.size(0)):
            src_b = src[b:b+1]                      # (1, S)
            enc, src_kpm = self.encode(src_b)        # (1, S, D)

            # Each beam: (score, token_ids)
            beams: list[tuple[float, list[int]]] = [(0.0, [bos_id])]
            completed: list[tuple[float, list[int]]] = []

            for _ in range(max_new_tokens):
                if not beams:
                    break
                # Batch all active beams
                tgt = torch.tensor([ids for _, ids in beams], dtype=torch.long, device=device)
                B2 = tgt.size(0)
                enc_exp = enc.expand(B2, -1, -1)
                kpm_exp = src_kpm.expand(B2, -1)

                h = self.decode(tgt, enc_exp, kpm_exp)
                logits = self.lm_head(h[:, -1])             # (B2, V)
                log_probs = torch.log_softmax(logits, dim=-1)

                # Expand each beam with top-beam_size tokens
                top_lp, top_ids = log_probs.topk(beam_size, dim=-1)  # (B2, beam)

                candidates: list[tuple[float, list[int]]] = []
                for i, (score, ids) in enumerate(beams):
                    for j in range(beam_size):
                        new_score = score + float(top_lp[i, j].item())
                        new_ids = ids + [int(top_ids[i, j].item())]
                        candidates.append((new_score, new_ids))

                # Keep top beam_size by length-normalized score
                def _normed(sc, ids):
                    lp = ((5 + len(ids)) / 6) ** length_penalty
                    return sc / lp

                candidates.sort(key=lambda x: _normed(x[0], x[1]), reverse=True)

                beams = []
                for score, ids in candidates:
                    if ids[-1] == eos_id:
                        completed.append((score, ids[:-1]))  # drop EOS
                    else:
                        beams.append((score, ids))
                    if len(beams) == beam_size and len(completed) + len(beams) >= beam_size:
                        break

                beams = beams[:beam_size]

                if len(completed) >= beam_size:
                    break

            if not completed:
                completed = beams if beams else [(0.0, [bos_id])]

            best = max(completed, key=lambda x: _normed(x[0], x[1]))
            results.append(best[1][1:])  # strip leading BOS

        # Pad to same length
        max_len = max((len(r) for r in results), default=1)
        out = torch.full((len(results), max_len), self.pad_id, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            out[i, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        return out

    @torch.no_grad()
    def generate_greedy(
        self, src: torch.Tensor, bos_id: int, eos_id: int,
        max_new_tokens: int = 128,
    ) -> torch.Tensor:
        self.eval()
        B = src.size(0)
        enc, src_kpm = self.encode(src)
        tgt = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_new_tokens):
            h = self.decode(tgt, enc, src_kpm)
            logits = self.lm_head(h[:, -1])
            next_tok = logits.argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, self.pad_id), next_tok)
            tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_id)
            if bool(finished.all().item()):
                break
        return tgt  # (B, T), includes leading BOS
