"""Tokenizer utilities for WMT14 En-De encoder-decoder MT.

Provides:
  - build_mt_tokenizer(): loads tokenizer by name, with special handling for
    IndicBART to avoid the transformers 5.6+ fast-tokenizer bug that drops
    Devanagari/Bengali vowel marks.
  - get_token_ids(): returns pad/bos/eos IDs.
"""

from __future__ import annotations

import json
from pathlib import Path


class _IndicBARTTokenizer:
    """Raw SentencePiece wrapper that avoids the AlbertTokenizer fast-path bug.

    transformers >= 5.6 reconstructs AlbertTokenizer via the *tokenizers* Rust
    library, which incorrectly drops dependent vowel signs (e.g. Devanagari
    matras) and produces completely different token IDs from the original
    spiece.model.  This wrapper calls sentencepiece directly and adds the
    language tokens (<2hi>, <2bn>, <2en>, etc.) manually.
    """

    PAD_ID = 0
    EOS_ID = 3
    BOS_ID = 2

    def __init__(self, spiece_path: str, added_tokens_path: str | None = None):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spiece_path)

        # Load added tokens (language IDs) from added_tokens.json
        self._added: dict[str, int] = {}
        if added_tokens_path and Path(added_tokens_path).exists():
            with open(added_tokens_path) as f:
                self._added = json.load(f)

        # Build reverse map for decoding
        self._id_to_added: dict[int, str] = {v: k for k, v in self._added.items()}

        # Special token IDs
        self.pad_token_id = self.PAD_ID
        self.eos_token_id = self.EOS_ID
        self.bos_token_id = self.BOS_ID
        self.pad_token = "<pad>"
        self.eos_token = "[SEP]"
        self.bos_token = "[CLS]"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _encode_text(self, text: str, add_special_tokens: bool) -> list[int]:
        """Tokenize a single string; language tokens are matched exactly."""
        out: list[int] = []
        for part in text.split(" "):
            if part in self._added:
                out.append(self._added[part])
            elif part:
                out.extend(self.sp.encode_as_ids(part))
        if add_special_tokens:
            out = [self.BOS_ID] + out + [self.EOS_ID]
        return out

    def _decode_ids(self, ids: list[int], skip_special_tokens: bool) -> str:
        """Decode a single list of IDs back to text."""
        pieces: list[str] = []
        for i in ids:
            if skip_special_tokens and i in (
                self.PAD_ID,
                self.EOS_ID,
                self.BOS_ID,
            ):
                continue
            if i in self._id_to_added:
                if not skip_special_tokens:
                    pieces.append(self._id_to_added[i])
                continue
            pieces.append(self.sp.id_to_piece(i))
        # SentencePiece uses '▁' to mark word beginnings; replace with space
        text = "".join(pieces).replace("▁", " ").strip()
        return text

    # ------------------------------------------------------------------
    # Public API used by src/eval.py
    # ------------------------------------------------------------------
    def __call__(
        self,
        texts: str | list[str],
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        padding: bool = True,
        return_tensors: str | None = None,
    ) -> dict:
        """Mimic transformers tokenizer __call__ for batch encoding."""
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self._encode_text(t, add_special_tokens) for t in texts]

        if max_length is not None and truncation:
            encoded = [e[:max_length] for e in encoded]

        if padding:
            max_len = max(len(e) for e in encoded)
            attention_mask = []
            for e in encoded:
                pad_len = max_len - len(e)
                e.extend([self.PAD_ID] * pad_len)
                attention_mask.append([1] * (max_len - pad_len) + [0] * pad_len)
        else:
            attention_mask = [[1] * len(e) for e in encoded]

        import torch

        result = {
            "input_ids": torch.tensor(encoded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        return result

    def batch_decode(
        self, token_lists: list[list[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        return [self._decode_ids(ids, skip_special_tokens) for ids in token_lists]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._decode_ids(ids, skip_special_tokens)

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in self._added:
            return self._added[token]
        # Try sentencepiece
        sp_id = self.sp.piece_to_id(token)
        if sp_id == self.sp.unk_id():
            # Return unk_id for unknown
            return sp_id
        return sp_id


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_mt_tokenizer(name_or_path: str = "Helsinki-NLP/opus-mt-en-de"):
    """Load a tokenizer for MT.

    For IndicBART we bypass the broken fast AlbertTokenizer and use raw
    SentencePiece directly.
    """
    from transformers import AutoTokenizer

    # IndicBART needs the custom wrapper
    if "indicbart" in name_or_path.lower():
        # Find the cached spiece.model
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache_root / f"models--{name_or_path.replace('/', '--')}"
        snapshots = list((model_dir / "snapshots").glob("*"))
        if not snapshots:
            raise RuntimeError(
                f"IndicBART cache not found at {model_dir}. "
                "Run AutoTokenizer.from_pretrained once to download."
            )
        snapshot = snapshots[0]
        spiece = snapshot / "spiece.model"
        added = snapshot / "added_tokens.json"
        return _IndicBARTTokenizer(str(spiece), str(added) if added.exists() else None)

    # Everything else (Marian, etc.) uses standard AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name_or_path)
    except ImportError as e:
        if "protobuf" not in str(e).lower():
            raise
        print("[tokenizer] protobuf missing; falling back to slow tokenizer (use_fast=False)")
        tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_token_ids(tokenizer) -> dict:
    return {
        "pad_id": tokenizer.pad_token_id,
        "eos_id": tokenizer.eos_token_id,
        "bos_id": getattr(tokenizer, "bos_token_id", None) or tokenizer.pad_token_id,
    }


def verify_token_ids(tokenizer) -> None:
    ids = get_token_ids(tokenizer)
    assert ids["pad_id"] is not None, "pad_token_id is None"
    assert ids["eos_id"] is not None, "eos_token_id is None"
    pad_tok = tokenizer.pad_token or "None"
    eos_tok = tokenizer.eos_token or "None"
    print(f"{'Token':<12} {'ID':>6}")
    print(f"{'─────────':<12} {'───':>6}")
    print(f"{pad_tok:<12} {ids['pad_id']:>6}")
    print(f"{eos_tok:<12} {ids['eos_id']:>6}")
    print(f"{'bos (dec)':<12} {ids['bos_id']:>6}")
