"""Step 4 — Full metric suite on the winning PE type's checkpoint.

Runs greedy + beam-5 decoding and computes BLEU + chrF + TER + BERTScore + COMET
on FLORES-200 devtest. This is the ONE place where BERTScore and COMET run —
the 3 main ablation runs in step3 stay fast by skipping them.

Usage:
    !python -m pipeline.step4_final_eval \\
      --checkpoint /content/drive/MyDrive/neur/outputs/checkpoints/asrope/best.pt \\
      --run-name asrope_final
"""

from __future__ import annotations

import argparse
import json

import torch

from pipeline import paths
from src.eval import compare_decoding


def _detect_device() -> str:
    if torch.cuda.is_available():
        print(f"[step4] device=cuda ({torch.cuda.get_device_name(0)})")
        return "cuda"
    print("[step4] device=cpu")
    return "cpu"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 4: full metric suite on winning PE")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt for the winning PE")
    parser.add_argument("--run-name", required=True, help="Subdir under outputs/metrics for this eval")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    if not paths.RAW_FLORES.exists():
        raise SystemExit(f"[step4] FLORES eval TSV missing at {paths.RAW_FLORES}. Run Step 1 first.")

    device = _detect_device()
    output_dir = paths.METRICS_DIR / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step4] FULL eval on {args.checkpoint}")
    print(f"[step4] output dir: {output_dir}")
    results = compare_decoding(
        checkpoint_path=args.checkpoint,
        eval_tsv=paths.RAW_FLORES,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        device=device,
        beam_sizes=[args.beam_size],
        metrics_mode="full",
    )

    print("\n" + "=" * 50)
    print(f"Final metrics for {args.run_name}")
    print("=" * 50)
    for decoding_label, m in results.items():
        print(f"\n{decoding_label}:")
        for k in ("bleu", "chrf", "ter", "bertscore_f1", "comet"):
            v = m.get(k)
            if v is not None:
                print(f"  {k:<14} {v:.4f}")
            else:
                print(f"  {k:<14} N/A")

    summary = {
        "checkpoint": args.checkpoint,
        "run_name": args.run_name,
        "results": results,
    }
    (output_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n[step4] wrote {output_dir / 'final_summary.json'}")


if __name__ == "__main__":
    main()
