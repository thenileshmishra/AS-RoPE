"""Tokenizer utilities for WMT14 En-De encoder-decoder MT."""

from __future__ import annotations


def build_mt_tokenizer(name_or_path: str = "Helsinki-NLP/opus-mt-en-de"):
    """Load a tokenizer for En-De MT. Pad token fallback to EOS if missing."""
    from transformers import AutoTokenizer

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
