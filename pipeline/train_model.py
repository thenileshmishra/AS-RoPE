"""Train encoder-decoder Transformer on WMT14 En-De with RoPE or Adaptive RoPE.

Usage:
    python -m pipeline.train_model --run-name adaptiverope_wmt14 \
        --pe-type adaptiverope \
        --tokenized-train processed_data_wmt14/tokenized/train_wmt14_en_de.pt \
        --tokenized-val   processed_data_wmt14/tokenized/val_wmt14_en_de.pt \
        --num-steps 30000 --eval-every 1000 --batch-size 512 --use-checkpoint
"""

from __future__ import annotations

import argparse
import json

import torch

from pipeline import paths
from src.train import TrainConfig, train


def _detect_device() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", 0) / 1e9
        print(f"[train] device=cuda ({name}, {mem:.1f} GB)")
        return "cuda"
    print("[train] device=cpu")
    return "cpu"


def _add_bool(parser, name, default, help_text):
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true",
                        default=default, help=help_text)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false",
                        help=f"disable --{name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train encoder-decoder with RoPE or Adaptive RoPE")

    parser.add_argument("--pe-type", choices=["rope", "adaptiverope", "sinusoidal"],
                        default="rope", help="Positional encoding type")
    parser.add_argument("--run-name", required=True,
                        help="Subdir under outputs/ for this run")

    parser.add_argument("--tokenized-train", default=str(paths.TOKENIZED_TRAIN))
    parser.add_argument("--tokenized-val", default=str(paths.TOKENIZED_VAL))

    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--n-enc-layers", type=int, default=6)
    parser.add_argument("--n-dec-layers", type=int, default=6)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--val-subset", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    _add_bool(parser, "use-bf16", True, "bf16 autocast")
    _add_bool(parser, "use-compile", True, "torch.compile")
    _add_bool(parser, "pin-memory", True, "pinned memory")
    _add_bool(parser, "checkpoint-light", True, "best.pt stores weights only")
    _add_bool(parser, "tie-embeddings", True, "tie src/tgt embedding + lm_head")
    _add_bool(parser, "use-checkpoint", False, "gradient checkpointing (saves ~50% memory, ~30% slower)")
    parser.add_argument("--dataloader-workers", type=int, default=2)

    args = parser.parse_args(argv)

    paths.ensure_dirs()
    device = _detect_device()

    ckpt_dir = paths.CHECKPOINT_DIR / args.run_name
    logs_dir = paths.LOGS_DIR / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        tokenized_train=args.tokenized_train,
        tokenized_val=args.tokenized_val,
        checkpoint_dir=str(ckpt_dir),
        logs_dir=str(logs_dir),
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        pe_type=args.pe_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_steps=args.num_steps,
        eval_every=args.eval_every,
        warmup_ratio=args.warmup_ratio,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum,
        val_subset_size=args.val_subset,
        seed=args.seed,
        device=device,
        use_bf16=args.use_bf16,
        use_compile=args.use_compile,
        dataloader_workers=args.dataloader_workers,
        pin_memory=args.pin_memory,
        tie_embeddings=args.tie_embeddings,
        use_checkpoint=args.use_checkpoint,
        checkpoint_light=args.checkpoint_light,
    )

    eff = cfg.batch_size * cfg.grad_accum_steps
    print(f"[train] pe={cfg.pe_type} d={cfg.d_model} heads={cfg.n_heads} "
          f"ff={cfg.d_ff} enc={cfg.n_enc_layers} dec={cfg.n_dec_layers} "
          f"seq={cfg.max_seq_len} batch={cfg.batch_size}x{cfg.grad_accum_steps}={eff} "
          f"steps={cfg.num_steps}")

    summary = train(cfg)
    print(f"[train] summary:\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
