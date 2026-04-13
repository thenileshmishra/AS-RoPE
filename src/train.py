"""Training loop for the GPT MT model with A100 optimizations.

Supports: BF16 mixed precision, torch.compile, Flash Attention (via SDPA),
gradient accumulation, and multi-worker DataLoaders.
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
from torch.utils.data import DataLoader

from src.dataset import collate_mt_batch, load_pairs
from src.dataset_v2 import MTDatasetV2
from src.model import GPT
from src.tokenizer_utils import build_mt_tokenizer


@dataclass
class TrainConfig:
    train_file: str
    val_file: str
    checkpoint_dir: str
    logs_dir: str
    tokenizer: str = "ai4bharat/IndicBART"
    max_seq_len: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_steps: int = 100_000
    eval_every: int = 5000
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"
    # --- A100 optimizations ---
    use_bf16: bool = False
    use_compile: bool = False
    grad_accum_steps: int = 1
    dataloader_workers: int = 0
    pin_memory: bool = False
    pe_type: str = "sinusoidal"


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


def _save_checkpoint(
    path: Path, model, optimizer, scheduler, step: int, config: dict, val_loss: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Handle torch.compile wrapper
    state_dict = model.state_dict()
    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "val_loss": val_loss,
        },
        path,
    )


def _enable_flash_attention(model) -> None:
    """Replace manual attention with F.scaled_dot_product_attention (Flash Attention)."""
    if not (torch.cuda.is_available() and hasattr(F, "scaled_dot_product_attention")):
        return

    # Get blocks from model or compiled model
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        orig = getattr(model, "_orig_mod", None)
        if orig is not None:
            blocks = getattr(orig, "blocks", None)
    if blocks is None:
        return

    for block in blocks:
        attn = block.attn
        original_forward = attn.forward

        def make_flash_forward(a):
            def flash_forward(x: torch.Tensor) -> torch.Tensor:
                bsz, seq_len, _ = x.shape
                q = a.q_proj(x).view(bsz, seq_len, a.n_heads, a.head_dim).transpose(1, 2)
                k = a.k_proj(x).view(bsz, seq_len, a.n_heads, a.head_dim).transpose(1, 2)
                v = a.v_proj(x).view(bsz, seq_len, a.n_heads, a.head_dim).transpose(1, 2)
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                out = out.transpose(1, 2).contiguous().view(bsz, seq_len, a.d_model)
                return a.out_proj(out)
            return flash_forward

        attn.forward = make_flash_forward(attn)

    print("[train] Flash Attention (SDPA) enabled")


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # TF32 for A100
    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    checkpoint_dir = Path(cfg.checkpoint_dir)
    logs_dir = Path(cfg.logs_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer (v2 with dedicated SEP token)
    tokenizer = build_mt_tokenizer(cfg.tokenizer)
    pad_id = tokenizer.pad_token_id
    vocab_size = len(tokenizer)

    train_pairs = load_pairs(cfg.train_file)
    val_pairs = load_pairs(cfg.val_file)
    if not train_pairs:
        raise ValueError(f"No training pairs found in {cfg.train_file}")
    print(f"[train] train_pairs={len(train_pairs):,} val_pairs={len(val_pairs):,}")
    print(f"[train] vocab_size={vocab_size} (tokenizer={cfg.tokenizer})")

    train_ds = MTDatasetV2(train_pairs, tokenizer, cfg.max_seq_len)
    val_ds = MTDatasetV2(val_pairs, tokenizer, cfg.max_seq_len)

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

    # Model
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

    # torch.compile
    if cfg.use_compile:
        try:
            model = torch.compile(model)
            print("[train] torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile failed ({e}), continuing without")

    # Flash Attention
    _enable_flash_attention(model)

    # Optimizer & scheduler
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
    run_config["pe_type"] = cfg.pe_type
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

    with open(metrics_path, "w", encoding="utf-8") as metrics_fp:
        for step in range(1, cfg.num_steps + 1):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)
            attention_mask = batch["attention_mask"].to(cfg.device)
            total_tokens_seen += int(attention_mask.sum().item())

            # Zero grad only at accumulation boundary
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

            # Step only at accumulation boundary
            if step % cfg.grad_accum_steps == 0 or step == cfg.num_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()

            elapsed_sec = time.monotonic() - start
            train_loss = float(loss.item())
            lr = optimizer.param_groups[0]["lr"]
            row: dict = {
                "step": step,
                "train_loss": train_loss,
                "lr": lr,
                "elapsed_sec": elapsed_sec,
                "tokens_seen": total_tokens_seen,
                "effective_batch_size": effective_batch,
                "tokens_per_sec": total_tokens_seen / max(elapsed_sec, 1e-6),
            }
            if cfg.device == "cuda":
                row["gpu_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)

            if step % cfg.eval_every == 0 or step == cfg.num_steps:
                val_loss, val_ppl = evaluate(model, val_loader, cfg.device)
                row["val_loss"] = val_loss
                row["val_ppl"] = val_ppl
                last_val = val_loss
                print(
                    f"step {step:6d} | train {train_loss:.4f} | lr {lr:.6f} "
                    f"| val {val_loss:.4f} | ppl {val_ppl:.2f} "
                    f"| tok/s {row['tokens_per_sec']:.0f}"
                )
                if math.isfinite(val_loss) and val_loss < best_val:
                    best_val = val_loss
                    best_step = step
                    _save_checkpoint(
                        best_ckpt, model, optimizer, scheduler,
                        step, run_config, val_loss,
                    )
                    best_saved = True
            else:
                if step % 100 == 0:
                    print(
                        f"step {step:6d} | train {train_loss:.4f} | lr {lr:.6f} "
                        f"| tok/s {row['tokens_per_sec']:.0f}"
                    )

            metrics_fp.write(json.dumps(row) + "\n")
            metrics_fp.flush()

    total_time = time.monotonic() - start
    _save_checkpoint(
        last_ckpt, model, optimizer, scheduler,
        cfg.num_steps, run_config, best_val,
    )
    if not best_saved:
        _save_checkpoint(
            best_ckpt, model, optimizer, scheduler,
            cfg.num_steps, run_config, best_val,
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
