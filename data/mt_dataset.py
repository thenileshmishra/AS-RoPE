"""Parallel Hindi-English machine translation dataset loader.

Supports TSV and JSONL input formats. Each sample is a source-target
sentence pair returned as raw strings or tokenized tensors.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


def normalize_text(text: str) -> str:
    """Normalize unicode to NFC and collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def load_pairs_tsv(path: str | Path) -> list[tuple[str, str]]:
    """Load source-target pairs from a TSV file (src\\ttgt per line)."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            parts = line.split("\t", maxsplit=1)
            src, tgt = normalize_text(parts[0]), normalize_text(parts[1])
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


def load_pairs_jsonl(path: str | Path) -> list[tuple[str, str]]:
    """Load source-target pairs from a JSONL file.

    Each line must be a JSON object with "src" and "tgt" keys.
    """
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = normalize_text(obj.get("src", ""))
            tgt = normalize_text(obj.get("tgt", ""))
            if src and tgt:
                pairs.append((src, tgt))
    return pairs


def load_pairs(path: str | Path) -> list[tuple[str, str]]:
    """Load pairs from TSV or JSONL based on file extension."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return load_pairs_jsonl(path)
    return load_pairs_tsv(path)


def deterministic_split(
    pairs: list[tuple[str, str]],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Split pairs into train/val/test using a deterministic hash-based assignment.

    Each pair is assigned to a split based on the MD5 hash of its source text,
    so the assignment is stable regardless of ordering.
    """
    train, val, test = [], [], []
    val_threshold = int(train_ratio * 1000)
    test_threshold = int((train_ratio + val_ratio) * 1000)

    for src, tgt in pairs:
        digest = hashlib.md5(f"{seed}:{src}".encode("utf-8")).hexdigest()
        bucket = int(digest[:3], 16) % 1000
        if bucket < val_threshold:
            train.append((src, tgt))
        elif bucket < test_threshold:
            val.append((src, tgt))
        else:
            test.append((src, tgt))

    return train, val, test


class MTDataset(Dataset):
    """Dataset of parallel source-target sentence pairs.

    If a tokenizer is provided, __getitem__ returns (src_ids, tgt_ids)
    as padded/truncated LongTensors. Otherwise returns (src_text, tgt_text).
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer=None,
        max_len: int = 128,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        src, tgt = self.pairs[idx]
        if self.tokenizer is None:
            return src, tgt

        src_enc = self.tokenizer(
            src,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_enc = self.tokenizer(
            tgt,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return src_enc["input_ids"].squeeze(0), tgt_enc["input_ids"].squeeze(0)


def build_mt_dataloaders(
    path: str | Path,
    batch_size: int = 16,
    tokenizer=None,
    max_len: int = 128,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load a parallel file and return train/val/test DataLoaders."""
    pairs = load_pairs(path)
    if not pairs:
        raise ValueError(f"No valid pairs found in {path}")
    train_pairs, val_pairs, test_pairs = deterministic_split(pairs, seed=seed)

    def make_loader(split_pairs: list[tuple[str, str]], shuffle: bool) -> DataLoader:
        ds = MTDataset(split_pairs, tokenizer=tokenizer, max_len=max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return (
        make_loader(train_pairs, shuffle=True),
        make_loader(val_pairs, shuffle=False),
        make_loader(test_pairs, shuffle=False),
    )


def create_sample_tsv(path: str | Path, n: int = 20) -> Path:
    """Write a tiny synthetic Hindi-English TSV for testing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hi_templates = [
        "यह एक परीक्षण वाक्य है।",
        "मुझे हिंदी पसंद है।",
        "आज मौसम अच्छा है।",
        "मैं स्कूल जा रहा हूँ।",
        "भारत एक बड़ा देश है।",
    ]
    en_templates = [
        "This is a test sentence.",
        "I like Hindi.",
        "The weather is nice today.",
        "I am going to school.",
        "India is a large country.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            hi = hi_templates[i % len(hi_templates)]
            en = en_templates[i % len(en_templates)]
            f.write(f"{hi}\t{en}\n")

    return path


if __name__ == "__main__":
    sample_path = create_sample_tsv(".cache/sample_mt.tsv", n=50)
    print(f"Created sample TSV: {sample_path}")

    pairs = load_pairs(sample_path)
    print(f"Loaded {len(pairs)} pairs")

    train, val, test = deterministic_split(pairs)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    ds = MTDataset(train)
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    print(f"Batch src[0]: {batch[0][0]}")
    print(f"Batch tgt[0]: {batch[1][0]}")
    print("Smoke test passed.")
