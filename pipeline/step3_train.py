"""Step 3 — Train the sinusoidal GPT on cleaned Samanantar and evaluate on FLORES-200.

Reads processed train/val TSVs from Google Drive, trains a decoder-only
transformer with sinusoidal positional encoding, saves checkpoints/logs to
Drive, and runs BLEU + chrF on the FLORES-200 devtest file also stored on
Drive.

Usage:
    !python -m pipeline.step3_train --num-steps 12000 --batch-size 16
"""

from __future__ import annotations

import argparse
import json

import torch

from pipeline import paths
from src.eval import evaluate_checkpoint
from src.train import TrainConfig, train


def _detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[step3] device=cuda ({name})")
        return "cuda"
    print("[step3] device=cpu")
    return "cpu"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 3: train + evaluate sinusoidal MT model")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=12000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-eval", action="store_true", help="Train only; skip FLORES-200 evaluation")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    if not paths.PROCESSED_TRAIN.exists() or not paths.PROCESSED_VAL.exists():
        raise SystemExit(
            f"[step3] processed splits missing under {paths.PROCESSED_DIR}. Run Step 2 first."
        )
    if not args.skip_eval and not paths.RAW_FLORES.exists():
        raise SystemExit(
            f"[step3] flores eval TSV missing at {paths.RAW_FLORES}. Run Step 1 first."
        )

    device = _detect_device()

    cfg = TrainConfig(
        train_file=str(paths.PROCESSED_TRAIN),
        val_file=str(paths.PROCESSED_VAL),
        checkpoint_dir=str(paths.CHECKPOINT_DIR),
        logs_dir=str(paths.LOGS_DIR),
        tokenizer=args.tokenizer,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        seed=args.seed,
        device=device,
    )
    print(f"[step3] config: {cfg}")
    summary = train(cfg)
    print(f"[step3] training summary: {json.dumps(summary, indent=2)}")

    if args.skip_eval:
        print("[step3] skipping FLORES-200 evaluation (--skip-eval)")
        return

    print(f"[step3] evaluating best checkpoint on {paths.RAW_FLORES}")
    metrics = evaluate_checkpoint(
        checkpoint_path=summary["best_ckpt"],
        eval_tsv=paths.RAW_FLORES,
        output_dir=paths.METRICS_DIR,
        max_new_tokens=args.max_new_tokens,
        device=device,
    )
    print(f"[step3] BLEU={metrics['bleu']:.2f}  chrF={metrics['chrf']:.2f}")
    print("[step3] done. All outputs saved under", paths.OUTPUTS_DIR)


if __name__ == "__main__":
    main()
