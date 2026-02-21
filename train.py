import argparse
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from urllib.request import urlretrieve

from model import GPT

GPT2_VOCAB_URL = "https://huggingface.co/gpt2/resolve/main/vocab.json"
GPT2_MERGES_URL = "https://huggingface.co/gpt2/resolve/main/merges.txt"


def load_wikitext103_train(root: str) -> str:
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", cache_dir=root)
    return "\n".join(dataset["text"])


def build_gpt2_bpe(cache_dir: str) -> ByteLevelBPETokenizer:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    vocab_path = cache_path / "vocab.json"
    merges_path = cache_path / "merges.txt"
    if not vocab_path.exists():
        urlretrieve(GPT2_VOCAB_URL, vocab_path)
    if not merges_path.exists():
        urlretrieve(GPT2_MERGES_URL, merges_path)

    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


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
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--use_wikitext", action="store_true")
    parser.add_argument("--data_root", type=str, default=".data")
    parser.add_argument("--tokenizer_cache", type=str, default=".cache/gpt2")
    parser.add_argument("--max_tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device

    if args.block_size > args.max_seq_len:
        raise ValueError("block_size must be <= max_seq_len")

    if args.use_wikitext:
        tokenizer = build_gpt2_bpe(args.tokenizer_cache)
        text = load_wikitext103_train(args.data_root)
        token_ids = tokenizer.encode(text).ids
        if args.max_tokens > 0:
            token_ids = token_ids[: args.max_tokens]
        args.vocab_size = tokenizer.get_vocab_size()
        data = torch.tensor(token_ids, device=device, dtype=torch.long)
    else:
        # Dummy deterministic token stream for a runnable minimal training loop.
        # Replace with your own tokenized corpus for real research runs.
        data_len = 200_000
        data = (torch.arange(data_len, device=device, dtype=torch.long) % args.vocab_size).long()

    model = GPT(
        vocab_size=args.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=args.max_seq_len,
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
                    "max_seq_len": args.max_seq_len,
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
