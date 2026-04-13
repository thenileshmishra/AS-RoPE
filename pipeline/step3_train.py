"""Step 3 — Train one PE variant on the pre-tokenized 1M subset.

Trains a decoder-only transformer (sinusoidal / rope / asrope / none) on the
pre-tokenized Hi->En subset produced by step2b, saves checkpoints and logs to
Drive, and runs a FAST greedy-only FLORES-200 evaluation (BLEU + chrF + TER).
COMET / BERTScore / beam-5 are intentionally skipped here and belong in
step4_final_eval.py, which runs once on the winning PE variant.

Recommended A100 invocation:
    !python -m pipeline.step3_train \\
      --pe-type sinusoidal \\
      --tokenized-train /content/tokenized/train_1m_hi_en.pt \\
      --tokenized-val   /content/tokenized/val_hi_en.pt
"""

from __future__ import annotations

import argparse
import json

import torch

from pipeline import paths
from src.eval import compare_decoding
from src.train import TrainConfig, train


def _detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, "total_memory", None) or getattr(props, "total_mem")
        print(f"[step3] device=cuda ({name}, {total_mem / 1e9:.1f} GB)")
        return "cuda"
    print("[step3] device=cpu")
    return "cpu"


def _add_bool(parser, name, default, help_text):
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true", default=default, help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"disable --{name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 3: train one PE variant + fast eval")

    # Model architecture
    parser.add_argument("--tokenizer", default="ai4bharat/IndicBART")
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument(
        "--pe-type",
        choices=["sinusoidal", "rope", "asrope", "none"],
        default="sinusoidal",
    )

    # Training hyperparameters (minimal-resource defaults)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=15_000)
    parser.add_argument("--eval-every", type=int, default=3_000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--val-subset", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    # A100 optimizations — ON BY DEFAULT
    _add_bool(parser, "use-bf16", True, "bf16 mixed precision")
    _add_bool(parser, "use-compile", True, "torch.compile")
    _add_bool(parser, "pin-memory", True, "pin dataloader memory")
    parser.add_argument("--dataloader-workers", type=int, default=2)
    _add_bool(parser, "checkpoint-light", True, "best.pt stores weights only")

    # Pre-tokenized cached dataset (preferred path)
    parser.add_argument(
        "--tokenized-train",
        default=None,
        help="Path to pre-tokenized train .pt (from step2b). If set, uses cached dataset.",
    )
    parser.add_argument(
        "--tokenized-val",
        default=None,
        help="Path to pre-tokenized val .pt (from step2b).",
    )

    # Output dir override (for per-PE subfolders on Drive)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional subdir under outputs/ — allows multiple PE runs to coexist.",
    )

    # Evaluation
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Train only; skip FLORES-200 evaluation",
    )

    args = parser.parse_args(argv)

    paths.ensure_dirs()

    use_cached = bool(args.tokenized_train and args.tokenized_val)
    if use_cached:
        train_file = args.tokenized_train
        val_file = args.tokenized_val
        print(f"[step3] using cached tokenized data: {train_file}")
    else:
        if not paths.PROCESSED_TRAIN.exists() or not paths.PROCESSED_VAL.exists():
            raise SystemExit(
                f"[step3] processed splits missing under {paths.PROCESSED_DIR}. "
                f"Run Step 2 (and optionally Step 2b) first."
            )
        train_file = str(paths.PROCESSED_TRAIN)
        val_file = str(paths.PROCESSED_VAL)

    if not args.skip_eval and not paths.RAW_FLORES.exists():
        raise SystemExit(
            f"[step3] FLORES eval TSV missing at {paths.RAW_FLORES}. Run Step 1 first."
        )

    device = _detect_device()

    # Per-run output folders so the 3 PE runs don't overwrite each other
    if args.run_name:
        ckpt_dir = paths.CHECKPOINT_DIR / args.run_name
        logs_dir = paths.LOGS_DIR / args.run_name
        metrics_dir = paths.METRICS_DIR / args.run_name
    else:
        ckpt_dir = paths.CHECKPOINT_DIR / args.pe_type
        logs_dir = paths.LOGS_DIR / args.pe_type
        metrics_dir = paths.METRICS_DIR / args.pe_type
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        train_file=train_file,
        val_file=val_file,
        checkpoint_dir=str(ckpt_dir),
        logs_dir=str(logs_dir),
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
        use_cached_dataset=use_cached,
        val_subset_size=args.val_subset,
        checkpoint_light=args.checkpoint_light,
    )

    eff_batch = cfg.batch_size * cfg.grad_accum_steps
    print(
        f"[step3] config: pe_type={cfg.pe_type} d_model={cfg.d_model} "
        f"n_layers={cfg.n_layers} batch={cfg.batch_size}x{cfg.grad_accum_steps}={eff_batch} "
        f"steps={cfg.num_steps} max_seq_len={cfg.max_seq_len} "
        f"bf16={cfg.use_bf16} compile={cfg.use_compile} cached={cfg.use_cached_dataset}"
    )

    summary = train(cfg)
    print(f"[step3] training summary: {json.dumps(summary, indent=2)}")

    if args.skip_eval:
        print("[step3] skipping FLORES-200 evaluation (--skip-eval)")
        return

    print(f"[step3] fast eval (greedy only) on {paths.RAW_FLORES}")
    comparison = compare_decoding(
        checkpoint_path=summary["best_ckpt"],
        eval_tsv=paths.RAW_FLORES,
        output_dir=metrics_dir,
        max_new_tokens=args.max_new_tokens,
        device=device,
        metrics_mode="fast",
    )

    greedy_m = comparison.get("greedy", {})
    print(
        f"[step3] greedy: BLEU={greedy_m.get('bleu', 0):.2f}  "
        f"chrF={greedy_m.get('chrf', 0):.2f}  TER={greedy_m.get('ter', 0):.2f}"
    )
    print("[step3] done. For the full metric suite on the winning PE, run step4_final_eval.")


if __name__ == "__main__":
    main()
