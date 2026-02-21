import argparse
import time
from pathlib import Path

import torch

from model import GPT


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, batch_size: int, block_size: int, device: str) -> float:
    model.eval()
    idx = torch.randint(0, data.size(0) - block_size - 1, (batch_size,), device=device)
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx])
    _, loss = model(x, y)
    model.train()
    return float(loss.item())


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    idx = torch.randint(0, data.size(0) - block_size - 1, (batch_size,), device=device)
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in idx])
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    # Dummy deterministic token stream for a runnable minimal training loop.
    # Replace with your own tokenized corpus for real research runs.
    data_len = 200_000
    data = (torch.arange(data_len, device=device, dtype=torch.long) % args.vocab_size).long()

    model = GPT(
        vocab_size=args.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=1024,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Training on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    start = time.time()
    train_losses: list[float] = []

    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch_size, args.block_size, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected at step {step}: {loss.item()}")

        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))

        if step % args.eval_interval == 0 or step == 1:
            eval_loss = estimate_loss(model, data, args.batch_size, args.block_size, device)
            elapsed = time.time() - start
            print(f"step {step:4d} | train_loss {loss.item():.4f} | eval_loss {eval_loss:.4f} | {elapsed:.1f}s")

        if step % args.save_interval == 0 or step == args.steps:
            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "vocab_size": args.vocab_size,
                    "d_model": 256,
                    "n_layers": 6,
                    "n_heads": 8,
                    "max_seq_len": 1024,
                },
            }
            torch.save(checkpoint, args.checkpoint_path)
            print(f"Checkpoint saved: {Path(args.checkpoint_path).resolve()}")

    if len(train_losses) >= 20:
        head = sum(train_losses[:10]) / 10
        tail = sum(train_losses[-10:]) / 10
        print(f"Training loss decreasing: {tail < head} (start_avg={head:.4f}, end_avg={tail:.4f})")


if __name__ == "__main__":
    main()
