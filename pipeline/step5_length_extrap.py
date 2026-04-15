"""Step 5 — Length extrapolation evaluation.

Trained at max_seq_len=192. Tests each checkpoint at 192, 256, 384 to measure
how well each PE generalises beyond its training length.

Metric: token-level perplexity on sequences of the target length (longer
sequences from the val set are used; shorter sequences are skipped).

Usage:
    !python -m pipeline.step5_length_extrap \
        --lang hi \
        --val /content/tokenized/val_hi_en.pt

    !python -m pipeline.step5_length_extrap \
        --lang bn \
        --val /content/drive/MyDrive/neur/processed_data_bn/tokenized/val_bn_en.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from pipeline import paths
from src.dataset_v2 import MTDatasetCached
from src.model import GPT


PE_TYPES = ["rope", "asrope", "asrope2", "asrope3"]
TEST_LENGTHS = [192, 256, 384]   # 192 = train length, 256/384 = extrapolation


@torch.no_grad()
def _ppl_at_length(
    model: GPT,
    dataset: MTDatasetCached,
    target_len: int,
    device: str,
    max_samples: int = 300,
) -> float:
    """Compute perplexity only on sequences that are >= target_len tokens.

    We truncate each sequence TO target_len so every sample contributes
    exactly the same number of positions — a fair comparison across lengths.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    counted = 0

    for idx in range(len(dataset)):
        item = dataset[idx]
        ids = item["input_ids"]      # 1-D tensor
        lbl = item["labels"]

        if len(ids) < target_len:
            continue                  # too short — skip

        # Truncate to exactly target_len
        ids = ids[:target_len].unsqueeze(0).to(device)
        lbl = lbl[:target_len].unsqueeze(0).to(device)

        # Temporarily expand model's max_seq_len if needed
        orig_max = model.max_seq_len
        if target_len > orig_max:
            model.max_seq_len = target_len
            # Also expand positional buffers if the PE uses them
            for block in model.blocks:
                rope = getattr(block.attn, "rope", None)
                if rope is not None and hasattr(rope, "positions"):
                    if len(rope.positions) < target_len:
                        rope.positions = torch.arange(
                            target_len, dtype=torch.float32, device=device
                        )

        try:
            logits, _ = model(ids)
        except Exception:
            model.max_seq_len = orig_max
            continue

        model.max_seq_len = orig_max

        valid_mask = lbl != -100
        if not valid_mask.any():
            continue

        loss = F.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            lbl.reshape(-1),
            ignore_index=-100,
        )
        n_valid = int(valid_mask.sum().item())
        total_loss += float(loss.item()) * n_valid
        total_tokens += n_valid
        counted += 1

        if counted >= max_samples:
            break

    if total_tokens == 0:
        return float("nan")
    avg = total_loss / total_tokens
    return math.exp(min(avg, 20.0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 5: length extrapolation eval")
    parser.add_argument("--lang", choices=["hi", "bn"], default="hi")
    parser.add_argument("--val", default=None,
                        help="Path to pre-tokenized val .pt (auto-detected if omitted)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=300,
                        help="Max sequences per (PE, length) cell — 300 is fast (~2 min total)")
    args = parser.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[step5] device={device} lang={args.lang}")

    if args.val:
        val_pt = Path(args.val)
    elif args.lang == "bn":
        val_pt = paths.PROCESSED_DIR_BN / "tokenized" / "val_bn_en.pt"
    else:
        val_pt = paths.PROCESSED_DIR / "tokenized" / "val_hi_en.pt"

    if not val_pt.exists():
        raise SystemExit(f"[step5] val file not found: {val_pt}")

    dataset = MTDatasetCached(str(val_pt))
    print(f"[step5] val dataset: {dataset.meta.get('n_examples', '?')} examples "
          f"(p99={dataset.meta.get('seq_len_p99', '?')} tokens)")

    results: dict[str, dict] = {}

    for pe in PE_TYPES:
        ckpt_path = paths.CHECKPOINT_DIR / f"{pe}_{args.lang}" / "best.pt"
        if not ckpt_path.exists():
            # Fall back to pe-type-only subdir (Hi-En runs)
            ckpt_path = paths.CHECKPOINT_DIR / pe / "best.pt"
        if not ckpt_path.exists():
            print(f"[step5] missing checkpoint for {pe} ({ckpt_path}) — skipping")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = GPT(
            vocab_size=int(cfg["vocab_size"]),
            d_model=int(cfg["d_model"]),
            n_layers=int(cfg["n_layers"]),
            n_heads=int(cfg["n_heads"]),
            max_seq_len=int(cfg["max_seq_len"]),
            pe_type=str(cfg.get("pe_type", "rope")),
            sep_id=cfg.get("sep_id"),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"\n[step5] {pe}")

        row: dict[str, float] = {}
        for tlen in TEST_LENGTHS:
            ppl = _ppl_at_length(model, dataset, tlen, device, args.max_samples)
            row[f"ppl_{tlen}"] = round(ppl, 2)
            print(f"  len={tlen:4d}  ppl={ppl:.2f}")
        results[pe] = row

    # Print comparison table
    print("\n" + "=" * 55)
    print(f"  Length Extrapolation — {args.lang.upper()}→En")
    print("=" * 55)
    header = f"{'PE':<12}" + "".join(f"{'ppl@'+str(l):>12}" for l in TEST_LENGTHS)
    print(header)
    print("-" * 55)
    for pe, row in results.items():
        line = f"{pe:<12}" + "".join(f"{row.get(f'ppl_{l}', float('nan')):>12.2f}"
                                      for l in TEST_LENGTHS)
        print(line)
    print("=" * 55)

    out_dir = paths.METRICS_DIR / f"length_extrap_{args.lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[step5] saved -> {out_path}")


if __name__ == "__main__":
    main()
