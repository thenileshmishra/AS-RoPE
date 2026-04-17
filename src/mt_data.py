"""MT pair dataset for encoder-decoder training.

Loads pre-tokenized flat int32 buffers with separate src/tgt offsets (zero-copy).
Collator:
  - pads src to max src length in batch
  - builds tgt_in  = [BOS, tgt_tokens]
  - builds tgt_out = [tgt_tokens, EOS]  with PAD positions masked to -100
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MTPairDatasetCached(Dataset):
    """Loads tokenized pairs saved as (src_flat, src_offsets, tgt_flat, tgt_offsets, meta)."""

    def __init__(self, path: str):
        blob = torch.load(path, weights_only=False)
        self.src_flat: torch.Tensor = blob["src_flat"]
        self.src_off: torch.Tensor = blob["src_offsets"]
        self.tgt_flat: torch.Tensor = blob["tgt_flat"]
        self.tgt_off: torch.Tensor = blob["tgt_offsets"]
        self.meta: dict = blob["meta"]

    def __len__(self) -> int:
        return self.src_off.numel() - 1

    def __getitem__(self, idx: int) -> dict:
        s0, s1 = int(self.src_off[idx].item()), int(self.src_off[idx + 1].item())
        t0, t1 = int(self.tgt_off[idx].item()), int(self.tgt_off[idx + 1].item())
        return {
            "src": self.src_flat[s0:s1].long(),
            "tgt": self.tgt_flat[t0:t1].long(),
        }


def _pad_to(seqs: list[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def build_collator(pad_id: int, bos_id: int, eos_id: int, label_ignore: int = -100):
    """Returns a collate_fn that produces src, tgt_in, tgt_out tensors."""
    def _collate(batch: list[dict]) -> dict:
        srcs = [b["src"] for b in batch]
        tgts = [b["tgt"] for b in batch]
        src = _pad_to(srcs, pad_id)

        tgt_in_seqs = [torch.cat([torch.tensor([bos_id], dtype=torch.long), t], dim=0) for t in tgts]
        tgt_out_seqs = [torch.cat([t, torch.tensor([eos_id], dtype=torch.long)], dim=0) for t in tgts]
        tgt_in = _pad_to(tgt_in_seqs, pad_id)
        tgt_out = _pad_to(tgt_out_seqs, pad_id)
        labels = tgt_out.clone()
        labels[labels == pad_id] = label_ignore
        return {"src": src, "tgt_in": tgt_in, "labels": labels}
    return _collate
