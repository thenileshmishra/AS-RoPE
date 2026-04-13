"""Training loop for the GPT MT model with A100 optimizations.

Features: BF16 mixed precision, torch.compile, native SDPA (Flash Attention),
gradient accumulation, multi-worker DataLoaders, tqdm progress bar, pre-tokenized
cached dataset path, subset validation, and light checkpoints for ablations.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.dataset import collate_mt_batch, load_pairs
from src.dataset_v2 import MTDatasetCached, MTDatasetV2
from src.model import GPT
from src.tokenizer_utils import build_mt_tokenizer


@dataclass
class TrainConfig:
    train_file: str
    val_file: str
    checkpoint_dir: str
    logs_dir: str
    tokenizer: str = "ai4bharat/IndicBART"
    max_seq_len: int = 192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_steps: int = 15_000
    eval_every: int = 3_000
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"
    # --- A100 optimizations (on by default) ---
    use_bf16: bool = True
    use_compile: bool = True
    grad_accum_steps: int = 4
    dataloader_workers: int = 2
    pin_memory: bool = True
    pe_type: str = "sinusoidal"
    # --- minimal-resource additions ---
    use_cached_dataset: bool = False  # True => load pre-tokenized .pt files
    val_subset_size: int = 500        # cap in-loop eval at this many examples
    checkpoint_light: bool = True     # best.pt stores weights only (no optim/sched)


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
def evaluate(model, loader: DataLoader, device: str) -> tuple[float, float]:
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


def _unwrap(model):
    return getattr(model, "_orig_mod", model)


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    step: int,
    config: dict,
    val_loss: float,
    light: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = _unwrap(model).state_dict()
    payload = {
        "step": step,
        "model_state_dict": state_dict,
        "config": config,
        "val_loss": val_loss,
    }
    if not light:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def _build_datasets(cfg: TrainConfig, tokenizer):
    """Return (train_ds, val_ds). Uses cached path if cfg.use_cached_dataset."""
    if cfg.use_cached_dataset:
        print(f"[train] loading cached tokenized train from {cfg.train_file}")
        train_ds = MTDatasetCached(cfg.train_file)
        print(f"[train] loading cached tokenized val from {cfg.val_file}")
        val_ds = MTDatasetCached(cfg.val_file)
        print(
            f"[train] cached meta: train={train_ds.meta.get('n_examples', '?')} "
            f"val={val_ds.meta.get('n_examples', '?')} "
            f"p99_len={train_ds.meta.get('seq_len_p99', '?')}"
        )
        return train_ds, val_ds

    train_pairs = load_pairs(cfg.train_file)
    val_pairs = load_pairs(cfg.val_file)
    if not train_pairs:
        raise ValueError(f"No training pairs found in {cfg.train_file}")
    print(f"[train] train_pairs={len(train_pairs):,} val_pairs={len(val_pairs):,}")
    train_ds = MTDatasetV2(train_pairs, tokenizer, cfg.max_seq_len)
    val_ds = MTDatasetV2(val_pairs, tokenizer, cfg.max_seq_len)
    return train_ds, val_ds


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    checkpoint_dir = Path(cfg.checkpoint_dir)
    logs_dir = Path(cfg.logs_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if cfg.use_cached_dataset:
        # Fast path: read vocab_size and pad_id from the cached meta —
        # no HF tokenizer load required.
        tokenizer = None
        train_ds, val_ds = _build_datasets(cfg, tokenizer)
        meta = train_ds.meta
        pad_id = int(meta["pad_id"])
        vocab_size = int(meta["vocab_size"])
        print(f"[train] vocab_size={vocab_size} pad_id={pad_id} (from cached meta)")
    else:
        tokenizer = build_mt_tokenizer(cfg.tokenizer)
        pad_id = tokenizer.pad_token_id
        vocab_size = len(tokenizer)
        print(f"[train] vocab_size={vocab_size} (tokenizer={cfg.tokenizer})")
        train_ds, val_ds = _build_datasets(cfg, tokenizer)

    if cfg.val_subset_size and len(val_ds) > cfg.val_subset_size:
        val_ds = Subset(val_ds, list(range(cfg.val_subset_size)))
        print(f"[train] capping in-loop val to {cfg.val_subset_size} examples")

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
        num_workers=cfg.dataloader_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.dataloader_workers > 0),
        prefetch_factor=(2 if cfg.dataloader_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=cfg.dataloader_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.dataloader_workers > 0),
        prefetch_factor=(2 if cfg.dataloader_workers > 0 else None),
    )

    model = GPT(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.max_seq_len,
        pe_type=cfg.pe_type,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params={n_params:,} pe_type={cfg.pe_type}")

    if cfg.use_compile:
        try:
            model = torch.compile(model)
            print("[train] torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile failed ({e}), continuing without")

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
    run_config["vocab_size"] = vocab_size
    run_config["n_params"] = n_params
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

    effective_batch = cfg.batch_size * cfg.grad_accum_steps

    pbar = tqdm(
        total=cfg.num_steps,
        desc=f"train[{cfg.pe_type}]",
        unit="step",
        dynamic_ncols=True,
    )

    with open(metrics_path, "w", encoding="utf-8") as metrics_fp:
        for step in range(1, cfg.num_steps + 1):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
            labels = batch["labels"].to(cfg.device, non_blocking=True)
            attention_mask = batch["attention_mask"]
            total_tokens_seen += int(attention_mask.sum().item())

            if (step - 1) % cfg.grad_accum_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=(cfg.use_bf16 and cfg.device == "cuda"),
            ):
                logits, _ = model(input_ids)
                loss = _masked_loss(logits, labels)

            scaled_loss = loss / cfg.grad_accum_steps
            scaled_loss.backward()

            if step % cfg.grad_accum_steps == 0 or step == cfg.num_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()

            elapsed_sec = time.monotonic() - start
            train_loss = float(loss.item())
            lr = optimizer.param_groups[0]["lr"]
            tok_per_sec = total_tokens_seen / max(elapsed_sec, 1e-6)

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{train_loss:.3f}",
                lr=f"{lr:.2e}",
                tok_s=f"{tok_per_sec:.0f}",
                best_val=f"{best_val:.3f}" if math.isfinite(best_val) else "—",
            )

            row: dict = {
                "step": step,
                "train_loss": train_loss,
                "lr": lr,
                "elapsed_sec": elapsed_sec,
                "tokens_seen": total_tokens_seen,
                "effective_batch_size": effective_batch,
                "tokens_per_sec": tok_per_sec,
            }
            if cfg.device == "cuda":
                row["gpu_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)

            if step % cfg.eval_every == 0 or step == cfg.num_steps:
                val_loss, val_ppl = evaluate(model, val_loader, cfg.device)
                row["val_loss"] = val_loss
                row["val_ppl"] = val_ppl
                last_val = val_loss
                pbar.write(
                    f"[eval] step {step:6d} | val {val_loss:.4f} | ppl {val_ppl:.2f} "
                    f"| tok/s {tok_per_sec:.0f}"
                )
                if math.isfinite(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    best_step = step
                    _save_checkpoint(
                        best_ckpt, model, optimizer, scheduler,
                        step, run_config, val_loss,
                        light=cfg.checkpoint_light,
                    )
                    best_saved = True

            metrics_fp.write(json.dumps(row) + "\n")
            metrics_fp.flush()

    pbar.close()

    total_time = time.monotonic() - start
    _save_checkpoint(
        last_ckpt, model, optimizer, scheduler,
        cfg.num_steps, run_config, best_val,
        light=False,  # last.pt always full — enables resume
    )
    if not best_saved:
        _save_checkpoint(
            best_ckpt, model, optimizer, scheduler,
            cfg.num_steps, run_config, best_val,
            light=cfg.checkpoint_light,
        )
        best_step = cfg.num_steps

    summary: dict = {
        "best_val_loss": best_val,
        "best_step": best_step,
        "final_val_loss": last_val,
        "total_steps": cfg.num_steps,
        "total_time_sec": total_time,
        "avg_step_sec": total_time / max(1, cfg.num_steps),
        "total_tokens_seen": total_tokens_seen,
        "tokens_per_sec_avg": total_tokens_seen / max(total_time, 1e-6),
        "pe_type": cfg.pe_type,
        "effective_batch_size": effective_batch,
        "n_params": n_params,
        "best_ckpt": str(best_ckpt),
        "last_ckpt": str(last_ckpt),
        "metrics_path": str(metrics_path),
    }
    if cfg.device == "cuda":
        summary["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
