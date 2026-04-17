"""Step 4 — Evaluate a checkpoint on WMT14 newstest2014 (greedy).

Metrics: BLEU, chrF, TER + BLEU-by-source-length.

Usage:
    python -m pipeline.step4_eval \
        --checkpoint outputs/checkpoints/rope_de/best.pt \
        --run-name   rope_de_eval
"""

from __future__ import annotations

import argparse

import torch

from pipeline import paths
from src.eval import evaluate_checkpoint


def _detect_device() -> str:
    if torch.cuda.is_available():
        print(f"[step4] device=cuda ({torch.cuda.get_device_name(0)})")
        return "cuda"
    print("[step4] device=cpu")
    return "cpu"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Eval a checkpoint on WMT14 newstest2014")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True,
                        help="Subdir under outputs/metrics for eval outputs")
    parser.add_argument("--eval-tsv", default=str(paths.RAW_WMT14_TEST))
    parser.add_argument("--tokenizer", default="Helsinki-NLP/opus-mt-en-de")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    from pathlib import Path
    if not Path(args.eval_tsv).exists():
        raise SystemExit(f"[step4] eval TSV missing at {args.eval_tsv}. Run step1_download first.")

    out_dir = paths.METRICS_DIR / args.run_name
    device = _detect_device()
    print(f"[step4] checkpoint: {args.checkpoint}")
    print(f"[step4] eval TSV : {args.eval_tsv}")
    print(f"[step4] output   : {out_dir}")

    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        eval_tsv=args.eval_tsv,
        output_dir=str(out_dir),
        device=device,
        tokenizer_name=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    ov = result["overall"]
    print("\n" + "=" * 50)
    print(f"Overall — {args.run_name}")
    print("=" * 50)
    print(f"  BLEU  {ov['bleu']:.2f}")
    print(f"  chrF  {ov['chrf']:.2f}")
    print(f"  TER   {ov['ter']:.2f}")
    print(f"  N     {ov['n']}")

    print("\nBLEU by source length:")
    print(f"  {'range':<10} {'n':>6} {'BLEU':>7} {'chrF':>7} {'TER':>7}")
    for b in result["by_src_length"]:
        n = b["n"]
        bleu = f"{b['bleu']:.2f}" if b["bleu"] is not None else "—"
        chrf = f"{b['chrf']:.2f}" if b["chrf"] is not None else "—"
        ter = f"{b['ter']:.2f}" if b["ter"] is not None else "—"
        print(f"  {b['range']:<10} {n:>6} {bleu:>7} {chrf:>7} {ter:>7}")


if __name__ == "__main__":
    main()
