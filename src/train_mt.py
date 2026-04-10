"""Day 3 Hindi-English MT training entrypoint.

Loads preprocessed train/val TSVs, builds next-token sequences with
target-only loss masking, trains the existing GPT model with a
configurable positional encoding, and saves best/last checkpoints
plus a JSONL metrics log.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.mt_dataset import load_pairs
from src.model import GPT


POS_ENC_CHOICES = ["rope", "alibi", "ntk_scaled_rope", "scaled_rope", "as_rope", "sinusoidal"]


# ── tokenizer ─────────────────────────────────────────────────────

def build_tokenizer(name_or_path: str):
    """Load a HuggingFace tokenizer. Fails loudly if unavailable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise RuntimeError(
            "transformers is required for training. Install with: pip install transformers"
        )
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── MT sequence construction with target-only masking ────────────

def build_mt_example(
    src: str,
    tgt: str,
    tokenizer,
    max_seq_len: int,
) -> dict:
    """Build one training example with target-only label masking.

    Sequence layout: ``[src_tokens, SEP, tgt_tokens, EOS]``
    - input_ids = sequence[:-1]
    - labels    = sequence[1:], with source positions replaced by -100.

    This is explicit and deterministic; see tests/test_train_mt_smoke.py.
    """
    sep_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    src_ids = tokenizer(src, add_special_tokens=False)["input_ids"]
    tgt_ids = tokenizer(tgt, add_special_tokens=False)["input_ids"]

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

    # Mask source segment in labels: labels[i] corresponds to predicting full[i+1].
    # Positions 0..len(src_ids)-1 in labels cover full[1..len(src_ids)],
    # which is src[1:] + SEP — all pre-target predictions. Mask them.
    src_len = len(src_ids)
    labels_masked = [-100] * src_len + labels[src_len:]

    return {
        "input_ids": input_ids,
        "labels": labels_masked,
        "src_len": src_len,
        "tgt_len": len(tgt_ids),
    }


def collate_mt_batch(
    examples: list[dict],
    pad_id: int,
) -> dict[str, torch.Tensor]:
    """Pad a batch of MT examples to the longest input in the batch.

    Output tensors:
      - input_ids: [B, L] (padded with pad_id)
      - labels:    [B, L] (padded with -100)
      - attention_mask: [B, L] (1 for real tokens, 0 for padding)
    """
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

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class MTTrainDataset(Dataset):
    """Dataset wrapping parallel pairs, tokenized on-the-fly per item."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_seq_len: int):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        src, tgt = self.pairs[idx]
        return build_mt_example(src, tgt, self.tokenizer, self.max_seq_len)


# ── loss, eval, scheduler helpers ────────────────────────────────

def compute_masked_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with -100 ignored (source positions)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: str) -> tuple[float, float]:
    """Return (val_loss, val_perplexity) over the loader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(input_ids)
        valid = int((labels != -100).sum().item())
        if valid == 0:
            continue
        loss = compute_masked_loss(logits, labels)
        total_loss += float(loss.item()) * valid
        total_tokens += valid
    model.train()
    if total_tokens == 0:
        return float("nan"), float("nan")
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step) / float(max(1, warmup))
    progress = float(step - warmup) / float(max(1, total - warmup))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: dict,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "val_loss": val_loss,
        },
        path,
    )


# ── training loop ────────────────────────────────────────────────

def train(args: argparse.Namespace) -> dict:
    """Run a single MT training job. Returns a summary dict."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = args.device
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Data
    tokenizer = build_tokenizer(args.tokenizer)
    pad_id = tokenizer.pad_token_id

    train_pairs = load_pairs(args.train_file)
    val_pairs = load_pairs(args.val_file)
    if not train_pairs:
        raise ValueError(f"No training pairs loaded from {args.train_file}")

    train_ds = MTTrainDataset(train_pairs, tokenizer, args.max_seq_len)
    val_ds = MTTrainDataset(val_pairs, tokenizer, args.max_seq_len)

    def _collate(batch):
        return collate_mt_batch(batch, pad_id)

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=_collate,
    )

    # Model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        positional_encoding=args.positional_encoding,
    ).to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    warmup_steps = max(1, int(0.1 * args.num_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: cosine_with_warmup(s, warmup_steps, args.num_steps),
    )

    # Run config (single source of truth for reloading)
    run_config = {
        "vocab_size": tokenizer.vocab_size,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "max_seq_len": args.max_seq_len,
        "positional_encoding": args.positional_encoding,
        "pad_token_id": pad_id,
        "tokenizer": args.tokenizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
    }
    (save_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    metrics_path = save_dir / "metrics.jsonl"
    best_ckpt_path = save_dir / "best.pt"
    last_ckpt_path = save_dir / "last.pt"

    best_val = float("inf")
    best_val_saved = False
    train_iter = _infinite(train_loader)
    model.train()

    with open(metrics_path, "w", encoding="utf-8") as metrics_fp:
        for step in range(1, args.num_steps + 1):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(input_ids)
            loss = compute_masked_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            train_loss = float(loss.item())
            row = {"step": step, "train_loss": train_loss, "lr": lr}

            if step % args.eval_every == 0 or step == args.num_steps:
                val_loss, val_ppl = evaluate(model, val_loader, device)
                row["val_loss"] = val_loss
                row["val_ppl"] = val_ppl
                print(
                    f"step {step:5d} | train_loss {train_loss:.4f} | "
                    f"lr {lr:.6f} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}"
                )
                if math.isfinite(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    _save_checkpoint(
                        best_ckpt_path, model, optimizer, scheduler, step, run_config, val_loss
                    )
                    best_val_saved = True
            else:
                print(f"step {step:5d} | train_loss {train_loss:.4f} | lr {lr:.6f}")

            metrics_fp.write(json.dumps(row) + "\n")
            metrics_fp.flush()

    # Always save a final "last" checkpoint
    _save_checkpoint(
        last_ckpt_path, model, optimizer, scheduler, args.num_steps, run_config, best_val
    )
    # If validation never produced a finite loss (e.g. empty val split),
    # fall back: the last checkpoint is also the best.
    if not best_val_saved:
        _save_checkpoint(
            best_ckpt_path, model, optimizer, scheduler, args.num_steps, run_config, best_val
        )

    return {
        "best_ckpt": str(best_ckpt_path),
        "last_ckpt": str(last_ckpt_path),
        "metrics_path": str(metrics_path),
        "run_config_path": str(save_dir / "run_config.json"),
        "best_val_loss": best_val,
    }


# ── CLI ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GPT for Hindi-English MT.")
    p.add_argument("--train-file", required=True, help="Processed train.tsv")
    p.add_argument("--val-file", required=True, help="Processed val.tsv")
    p.add_argument("--tokenizer", default="gpt2", help="HuggingFace tokenizer name/path")
    p.add_argument(
        "--positional-encoding",
        default="rope",
        choices=POS_ENC_CHOICES,
        help="Positional encoding variant",
    )
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--save-dir", default="results/mt_runs/default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    # Small-model knobs (optional, CPU-friendly defaults)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    print(
        f"[train_mt] train={args.train_file} val={args.val_file} "
        f"pe={args.positional_encoding} steps={args.num_steps} "
        f"bs={args.batch_size} lr={args.learning_rate} device={args.device}"
    )
    result = train(args)
    print(f"[train_mt] best ckpt: {result['best_ckpt']}")
    print(f"[train_mt] last ckpt: {result['last_ckpt']}")
    print(f"[train_mt] metrics : {result['metrics_path']}")
    print(f"[train_mt] best val_loss: {result['best_val_loss']:.4f}")
    return result


if __name__ == "__main__":
    main()
