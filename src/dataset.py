"""Parallel Hindi-English dataset loader with target-only masked sequences."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import torch
from torch.utils.data import Dataset


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def load_pairs(path: str | Path) -> list[tuple[str, str]]:
    """Load ``src\\ttgt`` pairs from a TSV file."""
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            src, tgt = line.split("\t", maxsplit=1)
            src, tgt = normalize_text(src), normalize_text(tgt)
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


def build_mt_example(src: str, tgt: str, tokenizer, max_seq_len: int) -> dict:
    """Build one training example with target-only label masking.

    Layout: ``[src, SEP, tgt, EOS]``. Positions inside src (and SEP itself)
    are masked to -100 so the loss only applies to target tokens.
    """
    sep_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    src_ids = tokenizer(src, add_special_tokens=False)["input_ids"]
    tgt_ids = tokenizer(tgt, add_special_tokens=False)["input_ids"]

    budget = max(2, max_seq_len - 2)
    if len(src_ids) + len(tgt_ids) > budget:
        half = budget // 2
        if len(src_ids) > half:
            src_ids = src_ids[:half]
        remaining = budget - len(src_ids)
        if len(tgt_ids) > remaining:
            tgt_ids = tgt_ids[:remaining]

    full = src_ids + [sep_id] + tgt_ids + [eos_id]
    input_ids = full[:-1]
    labels = full[1:]
    src_len = len(src_ids)
    labels = [-100] * src_len + labels[src_len:]
    return {"input_ids": input_ids, "labels": labels}


def collate_mt_batch(examples: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(e["input_ids"]) for e in examples)
    bsz = len(examples)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, ex in enumerate(examples):
        n = len(ex["input_ids"])
        input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class MTDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_seq_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        src, tgt = self.pairs[idx]
        return build_mt_example(src, tgt, self.tokenizer, self.max_seq_len)


def build_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
