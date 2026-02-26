import argparse

import torch

from model import GPT
from train import estimate_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device

    model = GPT(
        vocab_size=args.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        max_seq_len=1024,
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Same deterministic stream as train.py for a minimal runnable eval.
    data_len = 200_000
    data = (torch.arange(data_len, device=device, dtype=torch.long) % args.vocab_size).long()

    loss = estimate_loss(model, data, args.batch_size, args.block_size, device)
    print(f"Evaluation script runs | eval_loss={loss:.4f} | checkpoint_step={checkpoint.get('step', 'unknown')}")


if __name__ == "__main__":
    main()
