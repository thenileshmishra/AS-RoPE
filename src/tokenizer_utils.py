"""Tokenizer utilities for Hindi-English MT with proper SEP/EOS separation."""

from __future__ import annotations


def build_mt_tokenizer(name_or_path: str = "ai4bharat/IndicBART"):
    """Load a multilingual tokenizer and add a dedicated <sep> token.

    The <sep> token is guaranteed to have a different ID than <eos>,
    which fixes the ambiguity bug where the decoder could not distinguish
    "start translating" from "stop generating".
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"sep_token": "<sep>"})
    return tok


def get_token_ids(tokenizer) -> dict:
    """Return a dict of the three critical token IDs."""
    return {
        "pad_id": tokenizer.pad_token_id,
        "sep_id": tokenizer.sep_token_id,
        "eos_id": tokenizer.eos_token_id,
    }


def verify_token_ids(tokenizer) -> None:
    """Assert SEP != EOS and print a summary table.

    Raises AssertionError with a clear message if any check fails.
    """
    ids = get_token_ids(tokenizer)

    assert ids["sep_id"] is not None, (
        "sep_token_id is None — did you call build_mt_tokenizer() "
        "which adds <sep> as a special token?"
    )
    assert ids["sep_id"] != ids["eos_id"], (
        f"sep_token_id ({ids['sep_id']}) == eos_token_id ({ids['eos_id']}). "
        "The model will not be able to distinguish source/target boundary "
        "from end-of-sequence."
    )

    pad_tok = tokenizer.pad_token or "None"
    sep_tok = tokenizer.sep_token or "None"
    eos_tok = tokenizer.eos_token or "None"

    print(f"{'Token':<12} {'ID':>6}")
    print(f"{'──────────':<12} {'───':>6}")
    print(f"{pad_tok:<12} {ids['pad_id']:>6}")
    print(f"{sep_tok:<12} {ids['sep_id']:>6}")
    print(f"{eos_tok:<12} {ids['eos_id']:>6}")
