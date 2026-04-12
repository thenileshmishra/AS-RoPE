"""Step 3 — Train the GPT on cleaned Samanantar and evaluate on FLORES-200.

Reads processed train/val TSVs from Google Drive, trains a decoder-only
transformer (sinusoidal PE by default), saves checkpoints/logs to Drive,
and runs comprehensive MT evaluation (BLEU + chrF + TER + BERTScore + COMET)
on the FLORES-200 devtest file.

Recommended A100 invocation for 5M pairs:
    !python -m pipeline.step3_train \\
      --use-bf16 --use-compile --pin-memory --dataloader-workers 2 \\
      --grad-accum 8 --batch-size 64 --num-steps 100000

Hyperparameter rationale (reference):
    Effective batch:  64 * 8 = 512 sequences
    Steps per epoch:  5,000,000 / 512 ≈ 9,765 steps
    100K steps:       ≈ 10 epochs — good for convergence
    Max seq len:      256 — fits ~95% of Hindi-English pairs with IndicBART
    Model size:       512d / 12L ≈ 70M params — competitive for a PE study
    Learning rate:    5e-4 with cosine warmup (5% warmup = 5000 steps)
    BF16 + compile:   ~2.5x throughput vs FP32, ~140MB model memory
"""

from __future__ import annotations

import argparse
import json

import torch

from pipeline import paths
from src.eval import compare_decoding, evaluate_checkpoint_v2
from src.train import TrainConfig, train


def _detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, "total_memory", None)
        if total_mem is None:
            total_mem = getattr(props, "total_mem")
        mem_gb = total_mem / 1e9
        print(f"[step3] device=cuda ({name}, {mem_gb:.1f} GB)")
        return "cuda"
    print("[step3] device=cpu")
    return "cpu"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 3: train + evaluate MT model")

    # Model architecture
    parser.add_argument("--tokenizer", default="ai4bharat/IndicBART")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument(
        "--pe-type",
        choices=["sinusoidal", "learned", "rope", "alibi", "none"],
        default="sinusoidal",
    )

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    # A100 optimizations
    parser.add_argument("--use-bf16", action="store_true", default=False)
    parser.add_argument("--use-compile", action="store_true", default=False)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=False)

    # Evaluation
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Train only; skip FLORES-200 evaluation",
    )

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
        use_bf16=args.use_bf16,
        use_compile=args.use_compile,
        grad_accum_steps=args.grad_accum,
        dataloader_workers=args.dataloader_workers,
        pin_memory=args.pin_memory,
        pe_type=args.pe_type,
    )

    eff_batch = cfg.batch_size * cfg.grad_accum_steps
    print(f"[step3] config: pe_type={cfg.pe_type} d_model={cfg.d_model} "
          f"n_layers={cfg.n_layers} batch={cfg.batch_size}x{cfg.grad_accum_steps}={eff_batch} "
          f"steps={cfg.num_steps} bf16={cfg.use_bf16} compile={cfg.use_compile}")

    summary = train(cfg)
    print(f"[step3] training summary: {json.dumps(summary, indent=2)}")

    if args.skip_eval:
        print("[step3] skipping FLORES-200 evaluation (--skip-eval)")
        return

    # Comprehensive evaluation with decoding comparison
    print(f"[step3] evaluating best checkpoint on {paths.RAW_FLORES}")
    comparison = compare_decoding(
        checkpoint_path=summary["best_ckpt"],
        eval_tsv=paths.RAW_FLORES,
        output_dir=paths.METRICS_DIR,
        max_new_tokens=args.max_new_tokens,
        device=device,
        beam_sizes=[1, 5],
    )

    greedy_m = comparison.get("greedy", {})
    beam_m = comparison.get("beam_5", {})
    print(f"[step3] greedy: BLEU={greedy_m.get('bleu', 0):.2f}  "
          f"chrF={greedy_m.get('chrf', 0):.2f}")
    print(f"[step3] beam-5: BLEU={beam_m.get('bleu', 0):.2f}  "
          f"chrF={beam_m.get('chrf', 0):.2f}")
    print("[step3] done. All outputs saved under", paths.OUTPUTS_DIR)


if __name__ == "__main__":
    main()
