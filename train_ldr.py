import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from model import GPT
from long_distance_retrieval_dataset import LongDistanceRetrievalDataset


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = LongDistanceRetrievalDataset(num_samples=1, seq_len=seq_len)
    value_tokens = torch.randint(
        low=dataset.random_low,
        high=dataset.random_high_exclusive,
        size=(batch_size,),
        dtype=torch.long,
        device=device,
    )

    base = torch.randint(
        low=1,
        high=120,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )
    input_ids = base + (base >= value_tokens.unsqueeze(1)).long()

    first_half_end = max(1, seq_len // 2)
    needle_positions = torch.randint(0, first_half_end, (batch_size,), device=device)

    batch_indices = torch.arange(batch_size, device=device)
    input_ids[batch_indices, needle_positions] = dataset.NEEDLE_ID
    input_ids[batch_indices, needle_positions + 1] = value_tokens
    input_ids[:, -1] = dataset.QUERY_ID

    target_ids = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=device)
    target_ids[:, -1] = value_tokens
    return input_ids, target_ids


@torch.no_grad()
def evaluate_query_accuracy(
    model: GPT,
    seq_len: int,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0
    correct = 0

    for _ in range(num_batches):
        input_ids, target_ids = make_batch(batch_size=batch_size, seq_len=seq_len, device=device)
        logits, _ = model(input_ids, targets=None)

        query_positions = (target_ids != -100)
        pred_tokens = logits.argmax(dim=-1)
        correct += (pred_tokens[query_positions] == target_ids[query_positions]).sum().item()
        total += query_positions.sum().item()

    if total == 0:
        return 0.0
    return correct / total


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    set_seed(args.seed)

    model = GPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.model_max_seq_len,
        mlp_ratio=args.mlp_ratio,
        use_as_rope=args.use_as_rope,
        use_scaled_rope=args.use_scaled_rope,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print(f"device={device}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print(f"train_seq_len={args.train_seq_len} batch_size={args.batch_size}")

    start_time = time.time()
    step = 0
    epoch = 0
    best_acc_512 = 0.0
    last_train_loss = 0.0
    ckpt_path = Path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    while step < args.max_steps:
        epoch += 1
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for _ in range(args.steps_per_epoch):
            if step >= args.max_steps:
                break

            step += 1
            model.train()
            input_ids, target_ids = make_batch(
                batch_size=args.batch_size,
                seq_len=args.train_seq_len,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(input_ids, targets=None)
            loss = criterion(logits.reshape(-1, args.vocab_size), target_ids.reshape(-1))
            loss.backward()
            optimizer.step()

            last_train_loss = float(loss.item())
            epoch_loss_sum += last_train_loss
            epoch_loss_count += 1

            if step % args.log_interval == 0 or step == 1:
                elapsed = time.time() - start_time
                print(f"step={step:6d} train_loss={last_train_loss:.6f} elapsed={elapsed:.1f}s")

        acc_512 = evaluate_query_accuracy(
            model=model,
            seq_len=512,
            batch_size=args.eval_batch_size,
            num_batches=args.eval_batches,
            device=device,
        )
        epoch_train_loss = epoch_loss_sum / max(1, epoch_loss_count)
        elapsed = time.time() - start_time
        print(
            f"epoch={epoch:4d} step={step:6d} train_loss={epoch_train_loss:.6f} "
            f"acc512={acc_512:.4f} best_acc512={best_acc_512:.4f} elapsed={elapsed:.1f}s"
        )

        if acc_512 > best_acc_512:
            best_acc_512 = acc_512
            checkpoint = {
                "step": step,
                "epoch": epoch,
                "best_acc_512": best_acc_512,
                "last_train_loss": last_train_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "vocab_size": args.vocab_size,
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                    "mlp_ratio": args.mlp_ratio,
                    "max_seq_len": args.model_max_seq_len,
                    "use_as_rope": bool(args.use_as_rope),
                    "use_scaled_rope": bool(args.use_scaled_rope),
                },
                "task": {
                    "needle_token_id": 126,
                    "query_token_id": 127,
                    "train_seq_len": args.train_seq_len,
                },
            }
            torch.save(checkpoint, ckpt_path)
            print(f"checkpoint_saved={ckpt_path.resolve()} best_acc512={best_acc_512:.4f}")

        if acc_512 >= args.target_acc_512:
            print(f"early_stop: acc512={acc_512:.4f} >= target={args.target_acc_512:.4f}")
            break

    print(
        f"training_done step={step} max_steps={args.max_steps} "
        f"best_acc512={best_acc_512:.4f} last_train_loss={last_train_loss:.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--train_seq_len", type=int, default=512)
    parser.add_argument("--model_max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--target_acc_512", type=float, default=0.99)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint_ldr.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_as_rope", action="store_true")
    parser.add_argument("--use_scaled_rope", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())