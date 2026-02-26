import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from model import GPT


def _hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.getenv(key)
        if token:
            return token
    return None


def get_tokenizer(cache_dir: str) -> GPT2TokenizerFast:
    token = _hf_token()
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(
        "gpt2",
        token=token,
        cache_dir=str(cache_path),
    )
    return tokenizer


def _concat_text(examples: list[str]) -> str:
    cleaned = [line for line in examples if line and not line.isspace()]
    return "\n\n".join(cleaned)


def get_or_create_tokenized_streams(data_cache_dir: str, tokenizer: GPT2TokenizerFast) -> tuple[torch.Tensor, torch.Tensor]:
    cache_path = Path(data_cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    train_path = cache_path / "wikitext2_train_gpt2_ids.pt"
    val_path = cache_path / "wikitext2_val_gpt2_ids.pt"

    if train_path.exists() and val_path.exists():
        train_ids = torch.load(train_path, map_location="cpu")
        val_ids = torch.load(val_path, map_location="cpu")
        return train_ids.long(), val_ids.long()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = _concat_text(dataset["train"]["text"])
    val_text = _concat_text(dataset["validation"]["text"])

    train_ids = torch.tensor(
        tokenizer(train_text, add_special_tokens=False)["input_ids"],
        dtype=torch.long,
    )
    val_ids = torch.tensor(
        tokenizer(val_text, add_special_tokens=False)["input_ids"],
        dtype=torch.long,
    )

    torch.save(train_ids, train_path)
    torch.save(val_ids, val_path)
    return train_ids, val_ids


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    idx = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def cosine_with_warmup_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--freq_stats_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--metrics_path", type=str, default="")
    parser.add_argument(
        "--positional_encoding",
        type=str,
        default="rope",
        choices=["rope", "scaled_rope", "as_rope", "alibi", "ntk_scaled_rope"],
    )
    parser.add_argument("--use_as_rope", action="store_true")
    parser.add_argument("--use_scaled_rope", action="store_true")
    parser.add_argument("--data_cache", type=str, default=".cache/wikitext2_gpt2")
    parser.add_argument("--tokenizer_cache", type=str, default=".cache/hf_tokenizer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.use_as_rope and args.use_scaled_rope:
        raise ValueError("--use_as_rope and --use_scaled_rope cannot both be set")
    if args.use_as_rope:
        args.positional_encoding = "as_rope"
    elif args.use_scaled_rope:
        args.positional_encoding = "scaled_rope"

    torch.manual_seed(args.seed)
    device = args.device

    tokenizer = get_tokenizer(args.tokenizer_cache)
    train_data, val_data = get_or_create_tokenized_streams(args.data_cache, tokenizer)

    if train_data.numel() <= args.context_length + 1:
        raise ValueError("Training token stream is too short for the selected context_length.")

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=8,
        max_seq_len=args.context_length,
        positional_encoding=args.positional_encoding,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup_lambda(step, args.warmup_steps, args.max_steps),
    )

    print(f"Training on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"positional_encoding={args.positional_encoding}")

    start = time.time()
    train_losses: list[float] = []
    metrics_rows: list[tuple[int, float, float]] = []

    for step in range(1, args.max_steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at step {step}: {loss.item()}")

        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(float(loss.item()))

        if step % args.log_interval == 0 or step == 1:
            elapsed = time.time() - start
            print(
                f"step {step:5d} | train_loss {loss.item():.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.6e} | {elapsed:.1f}s"
            )
            metrics_rows.append((step, float(loss.item()), float(optimizer.param_groups[0]["lr"])))

        if args.positional_encoding == "as_rope" and model.freq_gates is not None and step % args.freq_stats_interval == 0:
            gates = model.freq_gates.detach()
            print(
                f"freq_gates step {step:5d} | mean {gates.mean().item():.6f} | std {gates.std(unbiased=False).item():.6f} | "
                f"min {gates.min().item():.6f} | max {gates.max().item():.6f}"
            )

        if args.positional_encoding == "scaled_rope" and model.gamma is not None and step % args.freq_stats_interval == 0:
            print(f"gamma step {step:5d} | value {model.gamma.detach().item():.6f}")

        if step % args.save_interval == 0 or step == args.max_steps:
            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": {
                    "vocab_size": tokenizer.vocab_size,
                    "d_model": 256,
                    "n_layers": 4,
                    "n_heads": 8,
                    "max_seq_len": args.context_length,
                    "positional_encoding": args.positional_encoding,
                    "use_as_rope": args.positional_encoding == "as_rope",
                    "use_scaled_rope": args.positional_encoding == "scaled_rope",
                },
            }
            torch.save(checkpoint, args.checkpoint_path)
            print(f"Checkpoint saved: {Path(args.checkpoint_path).resolve()}")

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "lr"])
            writer.writerows(metrics_rows)
        print(f"Metrics saved: {metrics_path.resolve()}")

    if val_data.numel() > args.context_length + 1:
        print(f"Validation token stream prepared: {val_data.numel()} tokens")

    if len(train_losses) >= 20:
        head = sum(train_losses[:10]) / 10
        tail = sum(train_losses[-10:]) / 10
        print(f"Training loss decreasing: {tail < head} (start_avg={head:.4f}, end_avg={tail:.4f})")


if __name__ == "__main__":
    main()
