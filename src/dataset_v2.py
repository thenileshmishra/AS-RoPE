"""V2 dataset module: fixes the SEP==EOS ambiguity and supports IndicBART tags.

Drop-in replacement for src.dataset. Re-exports unchanged utilities so that
downstream code can switch imports from ``src.dataset`` to ``src.dataset_v2``.
"""

from __future__ import annotations

from torch.utils.data import Dataset

# Re-export unchanged utilities from src.dataset
from src.dataset import MTDataset, collate_mt_batch, load_pairs, normalize_text  # noqa: F401
from src.tokenizer_utils import build_mt_tokenizer, get_token_ids


def build_mt_example_v2(src: str, tgt: str, tokenizer, max_seq_len: int) -> dict:
    """Build one training example with a dedicated SEP token (SEP != EOS).

    If the tokenizer has IndicBART language tags (``<2hi>``, ``<2en>``),
    they are prepended to improve translation direction signalling.

    Layout: ``[src_tokens] [SEP] [tgt_tokens] [EOS]``
    Labels: src_tokens and SEP are masked to -100; tgt_tokens and EOS carry loss.
    """
    sep_id = tokenizer.sep_token_id
    eos_id = tokenizer.eos_token_id

    # Optionally prepend IndicBART language tags
    hi_tag = "<2hi>" if "<2hi>" in tokenizer.get_vocab() else ""
    en_tag = "<2en>" if "<2en>" in tokenizer.get_vocab() else ""

    src_text = f"{hi_tag} {src}".strip() if hi_tag else src
    tgt_text = f"{en_tag} {tgt}".strip() if en_tag else tgt

    src_ids = tokenizer(src_text, add_special_tokens=False)["input_ids"]
    tgt_ids = tokenizer(tgt_text, add_special_tokens=False)["input_ids"]

    # Budget: leave room for SEP + EOS
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


class MTDatasetV2(Dataset):
    """Dataset using ``build_mt_example_v2`` (dedicated SEP + IndicBART tags)."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_seq_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        src, tgt = self.pairs[idx]
        return build_mt_example_v2(src, tgt, self.tokenizer, self.max_seq_len)


def build_tokenizer_v2(name_or_path: str = "ai4bharat/IndicBART"):
    """Thin wrapper around build_mt_tokenizer for naming symmetry."""
    return build_mt_tokenizer(name_or_path)


class MTDatasetCached(Dataset):
    """Reads pre-tokenized flat int32 tensors produced by step2b.

    Expects a torch.save dict with keys:
        input_ids_flat: int32 Tensor [total_tokens]
        labels_flat:    int32 Tensor [total_tokens]
        offsets:        int64 Tensor [n_examples + 1]
        meta:           dict

    __getitem__ slices the flat tensors — no tokenizer calls in the training hot path.
    """

    def __init__(self, tokenized_path):
        import torch

        blob = torch.load(tokenized_path, map_location="cpu", weights_only=False)
        self.input_ids_flat = blob["input_ids_flat"]
        self.labels_flat = blob["labels_flat"]
        self.offsets = blob["offsets"]
        self.meta = blob.get("meta", {})
        self._n = int(self.offsets.numel() - 1)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        start = int(self.offsets[idx].item())
        end = int(self.offsets[idx + 1].item())
        return {
            "input_ids": self.input_ids_flat[start:end],
            "labels": self.labels_flat[start:end],
        }
