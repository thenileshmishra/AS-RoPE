"""Training loop for the sinusoidal GPT MT model."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset import MTDataset, build_tokenizer, collate_mt_batch, load_pairs
from src.model import GPT


@dataclass
class TrainConfig:
    train_file: str
    val_file: str
    checkpoint_dir: str
    logs_dir: str
    tokenizer: str = "gpt2"
    max_seq_len: int = 128
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_steps: int = 12000
    eval_every: int = 1000
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"


def _cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _masked_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: str) -> tuple[float, float]:
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
        loss = _masked_loss(logits, labels)
        total_loss += float(loss.item()) * valid
        total_tokens += valid
    model.train()
    if total_tokens == 0:
        return float("nan"), float("nan")
    avg = total_loss / total_tokens
    return avg, math.exp(min(avg, 20.0))


def _infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _save_checkpoint(path: Path, model: GPT, step: int, config: dict, val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "config": config,
            "val_loss": val_loss,
        },
        path,
    )


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    logs_dir = Path(cfg.logs_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = build_tokenizer(cfg.tokenizer)
    pad_id = tokenizer.pad_token_id

    train_pairs = load_pairs(cfg.train_file)
    val_pairs = load_pairs(cfg.val_file)
    if not train_pairs:
        raise ValueError(f"No training pairs found in {cfg.train_file}")
    print(f"[train] train_pairs={len(train_pairs):,} val_pairs={len(val_pairs):,}")

    train_ds = MTDataset(train_pairs, tokenizer, cfg.max_seq_len)
    val_ds = MTDataset(val_pairs, tokenizer, cfg.max_seq_len)

    def _collate(batch):
        return collate_mt_batch(batch, pad_id)

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=_collate
    )

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.max_seq_len,
    ).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    warmup_steps = max(1, int(cfg.warmup_ratio * cfg.num_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_with_warmup(s, warmup_steps, cfg.num_steps),
    )

    run_config = asdict(cfg)
    run_config["vocab_size"] = tokenizer.vocab_size
    run_config["positional_encoding"] = "sinusoidal"
    (logs_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    metrics_path = logs_dir / "metrics.jsonl"
    best_ckpt = checkpoint_dir / "best.pt"
    last_ckpt = checkpoint_dir / "last.pt"
    summary_path = logs_dir / "run_summary.json"

    best_val = float("inf")
    best_step = -1
    best_saved = False
    last_val = float("nan")
    total_tokens_seen = 0
    train_iter = _infinite(train_loader)
    model.train()
    start = time.monotonic()

    with open(metrics_path, "w", encoding="utf-8") as metrics_fp:
        for step in range(1, cfg.num_steps + 1):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            total_tokens_seen += int(attention_mask.sum().item())

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(input_ids)
            loss = _masked_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            train_loss = float(loss.item())
            lr = optimizer.param_groups[0]["lr"]
            row = {
                "step": step,
                "train_loss": train_loss,
                "lr": lr,
                "elapsed_sec": time.monotonic() - start,
                "tokens_seen": total_tokens_seen,
            }

            if step % cfg.eval_every == 0 or step == cfg.num_steps:
                val_loss, val_ppl = evaluate(model, val_loader, cfg.device)
                row["val_loss"] = val_loss
                row["val_ppl"] = val_ppl
                last_val = val_loss
                print(
                    f"step {step:6d} | train {train_loss:.4f} | lr {lr:.6f} "
                    f"| val {val_loss:.4f} | ppl {val_ppl:.2f}"
                )
                if math.isfinite(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    best_step = step
                    _save_checkpoint(best_ckpt, model, step, run_config, val_loss)
                    best_saved = True
            else:
                print(f"step {step:6d} | train {train_loss:.4f} | lr {lr:.6f}")

            metrics_fp.write(json.dumps(row) + "\n")
            metrics_fp.flush()

    total_time = time.monotonic() - start
    _save_checkpoint(last_ckpt, model, cfg.num_steps, run_config, best_val)
    if not best_saved:
        _save_checkpoint(best_ckpt, model, cfg.num_steps, run_config, best_val)
        best_step = cfg.num_steps

    summary = {
        "best_val_loss": best_val,
        "best_step": best_step,
        "final_val_loss": last_val,
        "total_steps": cfg.num_steps,
        "total_time_sec": total_time,
        "avg_step_sec": total_time / max(1, cfg.num_steps),
        "total_tokens_seen": total_tokens_seen,
        "best_ckpt": str(best_ckpt),
        "last_ckpt": str(last_ckpt),
        "metrics_path": str(metrics_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
