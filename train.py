import argparse
import time

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
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    # Dummy integer token stream for a runnable minimal training loop.
    # Replace with your own tokenized corpus for real research runs.
    data_len = 200_000
    data = torch.randint(0, args.vocab_size, (data_len,), dtype=torch.long, device=device)

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
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch_size, args.block_size, device)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            eval_loss = estimate_loss(model, data, args.batch_size, args.block_size, device)
            elapsed = time.time() - start
            print(f"step {step:4d} | train_loss {loss.item():.4f} | eval_loss {eval_loss:.4f} | {elapsed:.1f}s")


if __name__ == "__main__":
    main()
