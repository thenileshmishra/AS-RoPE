import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from long_distance_retrieval_dataset import LongDistanceRetrievalDataset
from model import GPT


VARIANTS = ("rope", "scaled_rope", "as_rope")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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

    base = torch.randint(low=1, high=120, size=(batch_size, seq_len), dtype=torch.long, device=device)
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

        query_positions = target_ids != -100
        preds = logits.argmax(dim=-1)
        correct += (preds[query_positions] == target_ids[query_positions]).sum().item()
        total += query_positions.sum().item()

    if total == 0:
        return 0.0
    return correct / total


def build_model(variant: str, args: argparse.Namespace, device: torch.device) -> GPT:
    return GPT(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.model_max_seq_len,
        mlp_ratio=args.mlp_ratio,
        use_as_rope=(variant == "as_rope"),
        use_scaled_rope=(variant == "scaled_rope"),
    ).to(device)


def train_variant(variant: str, args: argparse.Namespace, device: torch.device) -> dict:
    model = build_model(variant, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_acc_512 = 0.0
    best_state_dict = None

    for step in range(1, args.max_steps + 1):
        model.train()
        input_ids, target_ids = make_batch(batch_size=args.batch_size, seq_len=args.train_seq_len, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(input_ids, targets=None)
        loss = criterion(logits.reshape(-1, args.vocab_size), target_ids.reshape(-1))
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1:
            acc_512 = evaluate_query_accuracy(
                model=model,
                seq_len=512,
                batch_size=args.eval_batch_size,
                num_batches=args.eval_batches,
                device=device,
            )
            print(
                f"[{variant}] step={step:6d} loss={loss.item():.6f} "
                f"acc512={acc_512:.4f} best={best_acc_512:.4f}"
            )
            if acc_512 > best_acc_512:
                best_acc_512 = acc_512
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if acc_512 >= args.target_acc_512:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    results = {
        "acc_512": evaluate_query_accuracy(model, 512, args.eval_batch_size, args.eval_batches, device),
        "acc_1024": evaluate_query_accuracy(model, 1024, args.eval_batch_size, args.eval_batches, device),
        "acc_2048": evaluate_query_accuracy(model, 2048, args.eval_batch_size, args.eval_batches, device),
        "acc_4096": evaluate_query_accuracy(model, 4096, args.eval_batch_size, args.eval_batches, device),
        "best_acc_512": best_acc_512,
    }

    if variant == "scaled_rope" and model.gamma is not None:
        results["gamma"] = float(model.gamma.detach().item())
    if variant == "as_rope" and model.freq_gates is not None:
        gates = model.freq_gates.detach().cpu()
        results["gate_min"] = float(gates.min().item())
        results["gate_max"] = float(gates.max().item())
        results["gate_mean"] = float(gates.mean().item())

    ckpt_path = Path(args.output_dir) / f"checkpoint_{variant}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": variant,
            "config": {
                "vocab_size": args.vocab_size,
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "mlp_ratio": args.mlp_ratio,
                "max_seq_len": args.model_max_seq_len,
                "use_as_rope": variant == "as_rope",
                "use_scaled_rope": variant == "scaled_rope",
            },
            "metrics": results,
        },
        ckpt_path,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate all three positional encoding variants.")
    parser.add_argument("--variants", type=str, default=",".join(VARIANTS))
    parser.add_argument("--output_dir", type=str, default="results_positional")

    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--train_seq_len", type=int, default=512)
    parser.add_argument("--model_max_seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_batches", type=int, default=20)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--target_acc_512", type=float, default=0.99)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in selected_variants if v not in VARIANTS]
    if invalid:
        raise ValueError(f"Invalid variants: {invalid}. Allowed: {VARIANTS}")

    all_results: dict[str, dict] = {}
    for variant in selected_variants:
        print(f"\n=== Running variant: {variant} ===")
        all_results[variant] = train_variant(variant, args, device)

    results_path = output_dir / "all_variants_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nFinal Results")
    print(json.dumps(all_results, indent=2))
    print(f"Saved: {results_path.resolve()}")


if __name__ == "__main__":
    main()
