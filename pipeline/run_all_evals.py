"""Run evaluation on all trained checkpoints + apply Position Interpolation (PI).

For each trained checkpoint, runs greedy + beam-5 evaluation.
For RoPE checkpoints, additionally evaluates with PI at multiple scales.

Usage:
    python -m pipeline.run_all_evals --lang de --beam-size 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pipeline import paths
from src.eval import evaluate_checkpoint
from src.positional import apply_position_interpolation
from src.eval import load_model_from_checkpoint


# Dataset configs
DATASETS = {
    "de": {
        "test_tsv": "raw_data/wmt14/test.tsv",
        "tokenizer": "Helsinki-NLP/opus-mt-en-de",
    },
    "hi": {
        "test_tsv": "raw_data/samanantar/test.tsv",
        "tokenizer": "Helsinki-NLP/opus-mt-en-de",
    },
    "bn": {
        "test_tsv": None,
        "tokenizer": "Helsinki-NLP/opus-mt-en-de",
    },
}


def eval_checkpoint(
    ckpt_path: str,
    run_name: str,
    test_tsv: str,
    tokenizer: str,
    device: str,
    beam_size: int = 5,
) -> dict:
    """Evaluate a single checkpoint."""
    out_dir = paths.METRICS_DIR / run_name
    if (out_dir / "eval_summary.json").exists():
        print(f"[eval] {run_name} already evaluated, skipping")
        return json.loads((out_dir / "eval_summary.json").read_text())

    return evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        eval_tsv=test_tsv,
        output_dir=str(out_dir),
        device=device,
        tokenizer_name=tokenizer,
        beam_size=beam_size,
    )


def eval_with_pi(
    ckpt_path: str,
    run_name: str,
    test_tsv: str,
    tokenizer: str,
    device: str,
    scales: list[float],
    beam_size: int = 5,
) -> list[dict]:
    """Evaluate a RoPE checkpoint with Position Interpolation at multiple scales."""
    results = []
    for scale in scales:
        pi_run_name = f"{run_name}_pi{scale}"
        out_dir = paths.METRICS_DIR / pi_run_name
        if (out_dir / "eval_summary.json").exists():
            print(f"[eval] {pi_run_name} already evaluated, skipping")
            results.append(json.loads((out_dir / "eval_summary.json").read_text()))
            continue

        model, cfg = load_model_from_checkpoint(ckpt_path, device)
        apply_position_interpolation(model, scale)
        print(f"[eval] Applied PI scale={scale} to {run_name}")

        # Save a temporary checkpoint with PI applied
        temp_ckpt = paths.CHECKPOINT_DIR / f"_temp_{pi_run_name}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": cfg,
        }, temp_ckpt)

        result = evaluate_checkpoint(
            checkpoint_path=str(temp_ckpt),
            eval_tsv=test_tsv,
            output_dir=str(out_dir),
            device=device,
            tokenizer_name=tokenizer,
            beam_size=beam_size,
        )
        result["pi_scale"] = scale
        results.append(result)
        temp_ckpt.unlink(missing_ok=True)

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--lang", choices=["de", "hi", "bn"], default="de")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--pi-scales", nargs="+", type=float, default=[1.5, 2.0, 3.0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    ds = DATASETS[args.lang]
    test_tsv = ds["test_tsv"]
    tokenizer = ds["tokenizer"]

    if test_tsv and not Path(test_tsv).exists():
        print(f"[eval] Warning: test TSV not found at {test_tsv}")
        return

    # Find all checkpoints for this language
    ckpt_dir = paths.CHECKPOINT_DIR
    all_checkpoints = []
    for subdir in ckpt_dir.iterdir():
        if not subdir.is_dir():
            continue
        if f"_{args.lang}" in subdir.name or subdir.name.endswith(f"_{args.lang}"):
            best_pt = subdir / "best.pt"
            if best_pt.exists():
                all_checkpoints.append((subdir.name, str(best_pt)))

    print(f"[eval] Found {len(all_checkpoints)} checkpoints for lang={args.lang}")

    for run_name, ckpt_path in sorted(all_checkpoints):
        print(f"\n[eval] {'='*60}")
        print(f"[eval] {run_name}")
        print(f"[eval] {'='*60}")

        # Standard eval
        eval_checkpoint(ckpt_path, run_name, test_tsv, tokenizer, args.device, args.beam_size)

        # PI eval for RoPE checkpoints
        if "rope" in run_name and "pi" not in run_name and "carope" not in run_name:
            print(f"[eval] Running PI variants for {run_name}...")
            eval_with_pi(ckpt_path, run_name, test_tsv, tokenizer, args.device, args.pi_scales, args.beam_size)

    print(f"\n[eval] All evaluations complete for lang={args.lang}")


if __name__ == "__main__":
    main()
