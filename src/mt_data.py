"""MT pair dataset for encoder-decoder training.

Loads pre-tokenized flat int32 buffers with separate src/tgt offsets (zero-copy).
Supports two formats:
  1. WMT14 format: src_flat, src_offsets, tgt_flat, tgt_offsets, meta
  2. Hi/Bn format: input_ids_flat, labels_flat, offsets, meta

Collator:
  - pads src to max src length in batch
  - builds tgt_in  = [BOS, tgt_tokens]
  - builds tgt_out = [tgt_tokens, EOS]  with PAD positions masked to -100
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class MTPairDatasetCached(Dataset):
    """Loads tokenized pairs saved as .pt files.

    Supports both WMT14 format (src/tgt separate) and unified format (input_ids/labels).
    """

    def __init__(self, path: str):
        blob = torch.load(path, weights_only=False)
        self.meta: dict = blob["meta"]

        # Detect format
        if "src_flat" in blob:
            # WMT14 format: separate src/tgt buffers
            self._mode = "wmt14"
            self.src_flat: torch.Tensor = blob["src_flat"]
            self.src_off: torch.Tensor = blob["src_offsets"]
            self.tgt_flat: torch.Tensor = blob["tgt_flat"]
            self.tgt_off: torch.Tensor = blob["tgt_offsets"]
        elif "input_ids_flat" in blob:
            # Unified format (Hi/Bn): single buffer with alternating src/tgt
            self._mode = "unified"
            self._flat: torch.Tensor = blob["input_ids_flat"]
            self._off: torch.Tensor = blob["offsets"]
        else:
            raise ValueError(
                f"Unknown format in {path}: expected src_flat/src_offsets or "
                f"input_ids_flat/offsets. Got keys: {list(blob.keys())}"
            )

    def __len__(self) -> int:
        if self._mode == "wmt14":
            return self.src_off.numel() - 1
        else:
            # Unified format: offsets has n+1 entries for n examples
            return self._off.numel() - 1

    def __getitem__(self, idx: int) -> dict:
        if self._mode == "wmt14":
            s0, s1 = int(self.src_off[idx].item()), int(self.src_off[idx + 1].item())
            t0, t1 = int(self.tgt_off[idx].item()), int(self.tgt_off[idx + 1].item())
            return {
                "src": self.src_flat[s0:s1].long(),
                "tgt": self.tgt_flat[t0:t1].long(),
            }
        else:
            # Unified format: each example is [src_tokens, sep_id, tgt_tokens]
            # We need to split on sep_id
            sep_id = int(self.meta.get("sep_id", 64014))
            o0, o1 = int(self._off[idx].item()), int(self._off[idx + 1].item())
            tokens = self._flat[o0:o1].long()
            # Find separator
            sep_positions = (tokens == sep_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                split = int(sep_positions[0].item())
                src = tokens[:split]
                tgt = tokens[split + 1:]
            else:
                # Fallback: no separator found, assume first half is src
                mid = len(tokens) // 2
                src = tokens[:mid]
                tgt = tokens[mid:]
            return {"src": src, "tgt": tgt}


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
