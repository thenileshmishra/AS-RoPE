"""Encoder-decoder MT training loop. Records val loss + training time.

Keys:
  - BF16 autocast on CUDA
  - optional torch.compile
  - cosine LR with warmup
  - gradient accumulation
  - periodic val loss; best checkpoint saved on best val loss
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

from src.model import EncoderDecoder
from src.mt_data import MTPairDatasetCached, build_collator


@dataclass
class TrainConfig:
    tokenized_train: str
    tokenized_val: str
    checkpoint_dir: str
    logs_dir: str
    d_model: int = 384
    n_heads: int = 6
    d_ff: int = 1536
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    max_seq_len: int = 256
    dropout: float = 0.1
    pe_type: str = "rope"
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    num_steps: int = 10_000
    eval_every: int = 1_000
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    seed: int = 42
    device: str = "cpu"
    use_bf16: bool = True
    use_compile: bool = True
    dataloader_workers: int = 2
    pin_memory: bool = True
    val_subset_size: int = 500
    tie_embeddings: bool = True
    use_checkpoint: bool = False
    checkpoint_light: bool = True


def _cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def _unwrap(model):
    return getattr(model, "_orig_mod", model)


@torch.no_grad()
def evaluate_loss(model, loader, device) -> tuple[float, float]:
    model.eval()
    total, n_tok = 0.0, 0
    for batch in loader:
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(src, tgt_in)
        valid = int((labels != -100).sum().item())
        if valid == 0:
            continue
        loss = _loss(logits, labels)
        total += float(loss.item()) * valid
        n_tok += valid
    model.train()
    if n_tok == 0:
        return float("nan"), float("nan")
    avg = total / n_tok
    return avg, math.exp(min(avg, 20.0))


def _save_ckpt(path: Path, model, optim, sched, step, config, val_loss, light=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": _unwrap(model).state_dict(),
        "config": config,
        "val_loss": val_loss,
    }
    if not light:
        payload["optimizer_state_dict"] = optim.state_dict()
        payload["scheduler_state_dict"] = sched.state_dict()
    torch.save(payload, path)


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    ckpt_dir = Path(cfg.checkpoint_dir)
    logs_dir = Path(cfg.logs_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MTPairDatasetCached(cfg.tokenized_train)
    val_ds = MTPairDatasetCached(cfg.tokenized_val)
    meta = train_ds.meta
    pad_id = int(meta["pad_id"])
    eos_id = int(meta["eos_id"])
    bos_id = int(meta["bos_id"])
    vocab_size = int(meta["vocab_size"])
    print(f"[train] pe={cfg.pe_type} vocab={vocab_size} pad={pad_id} bos={bos_id} eos={eos_id}")
    print(f"[train] train={len(train_ds):,}  val={len(val_ds):,}")

    if cfg.val_subset_size and len(val_ds) > cfg.val_subset_size:
        val_ds = Subset(val_ds, list(range(cfg.val_subset_size)))
        print(f"[train] capping in-loop val to {cfg.val_subset_size} examples")

    collate = build_collator(pad_id=pad_id, bos_id=bos_id, eos_id=eos_id)
    gen = torch.Generator(); gen.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
        generator=gen, num_workers=cfg.dataloader_workers, pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.dataloader_workers > 0),
        prefetch_factor=(2 if cfg.dataloader_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate,
        num_workers=cfg.dataloader_workers, pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.dataloader_workers > 0),
        prefetch_factor=(2 if cfg.dataloader_workers > 0 else None),
    )

    model = EncoderDecoder(
        vocab_size=vocab_size, pad_id=pad_id,
        d_model=cfg.d_model, n_heads=cfg.n_heads, d_ff=cfg.d_ff,
        n_enc_layers=cfg.n_enc_layers, n_dec_layers=cfg.n_dec_layers,
        max_seq_len=cfg.max_seq_len, pe_type=cfg.pe_type,
        dropout=cfg.dropout, tie_embeddings=cfg.tie_embeddings,
        use_checkpoint=cfg.use_checkpoint,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params={n_params:,}")

    if cfg.use_compile:
        try:
            model = torch.compile(model)
            print("[train] torch.compile enabled")
        except Exception as e:
            print(f"[train] torch.compile failed ({e})")

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.98),
    )
    warmup = max(1, int(cfg.warmup_ratio * cfg.num_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: _cosine(s, warmup, cfg.num_steps)
    )

    run_config = asdict(cfg)
    run_config["vocab_size"] = vocab_size
    run_config["pad_id"] = pad_id
    run_config["bos_id"] = bos_id
    run_config["eos_id"] = eos_id
    run_config["n_params"] = n_params
    (logs_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    metrics_path = logs_dir / "metrics.jsonl"
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    summary_path = logs_dir / "run_summary.json"

    best_val = float("inf")
    best_step = -1
    best_saved = False
    last_val = float("nan")
    total_tokens = 0
    it = _infinite(train_loader)
    model.train()
    start = time.monotonic()
    eff_batch = cfg.batch_size * cfg.grad_accum_steps

    pbar = tqdm(total=cfg.num_steps, desc=f"train[{cfg.pe_type}]", unit="step", dynamic_ncols=True)
    with open(metrics_path, "w") as fp:
        for step in range(1, cfg.num_steps + 1):
            batch = next(it)
            src = batch["src"].to(cfg.device, non_blocking=True)
            tgt_in = batch["tgt_in"].to(cfg.device, non_blocking=True)
            labels = batch["labels"].to(cfg.device, non_blocking=True)
            total_tokens += int((labels != -100).sum().item())

            if (step - 1) % cfg.grad_accum_steps == 0:
                optim.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(cfg.use_bf16 and cfg.device == "cuda")):
                logits = model(src, tgt_in)
                loss = _loss(logits, labels)

            (loss / cfg.grad_accum_steps).backward()

            if step % cfg.grad_accum_steps == 0 or step == cfg.num_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optim.step()
                sched.step()

            elapsed = time.monotonic() - start
            lr = optim.param_groups[0]["lr"]
            tok_s = total_tokens / max(elapsed, 1e-6)
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{float(loss.item()):.3f}", lr=f"{lr:.2e}",
                tok_s=f"{tok_s:.0f}",
                best=f"{best_val:.3f}" if math.isfinite(best_val) else "—",
            )

            row = {
                "step": step, "train_loss": float(loss.item()), "lr": lr,
                "elapsed_sec": elapsed, "tokens_seen": total_tokens,
                "effective_batch_size": eff_batch, "tokens_per_sec": tok_s,
            }
            if cfg.device == "cuda":
                row["gpu_mem_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)

            if step % cfg.eval_every == 0 or step == cfg.num_steps:
                v_loss, v_ppl = evaluate_loss(model, val_loader, cfg.device)
                row["val_loss"] = v_loss
                row["val_ppl"] = v_ppl
                last_val = v_loss
                pbar.write(f"[eval] step {step:6d} | val {v_loss:.4f} | ppl {v_ppl:.2f} | tok/s {tok_s:.0f}")
                if math.isfinite(v_loss) and v_loss < best_val:
                    best_val = v_loss
                    best_step = step
                    _save_ckpt(best_ckpt, model, optim, sched, step, run_config, v_loss,
                               light=cfg.checkpoint_light)
                    best_saved = True

            fp.write(json.dumps(row) + "\n")
            fp.flush()

    pbar.close()

    total_time = time.monotonic() - start
    _save_ckpt(last_ckpt, model, optim, sched, cfg.num_steps, run_config, best_val, light=False)
    if not best_saved:
        _save_ckpt(best_ckpt, model, optim, sched, cfg.num_steps, run_config, best_val,
                   light=cfg.checkpoint_light)
        best_step = cfg.num_steps

    summary = {
        "pe_type": cfg.pe_type,
        "best_val_loss": best_val,
        "best_step": best_step,
        "final_val_loss": last_val,
        "total_steps": cfg.num_steps,
        "total_time_sec": total_time,
        "avg_step_sec": total_time / max(1, cfg.num_steps),
        "total_tokens_seen": total_tokens,
        "tokens_per_sec_avg": total_tokens / max(total_time, 1e-6),
        "effective_batch_size": eff_batch,
        "n_params": n_params,
        "best_ckpt": str(best_ckpt),
        "last_ckpt": str(last_ckpt),
        "metrics_path": str(metrics_path),
    }
    if cfg.device == "cuda":
        summary["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary
