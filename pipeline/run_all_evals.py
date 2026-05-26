"""Run evaluation on all trained checkpoints + Position Interpolation (PI) variants.

For each trained checkpoint:
  1. Run greedy + beam-5 evaluation
  2. For RoPE checkpoints: additionally evaluate with PI at multiple scales

Auto-detects tokenizer from checkpoint config for cross-lingual compatibility.

Usage:
    python -m pipeline.run_all_evals
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


def eval_checkpoint(
    ckpt_path: str,
    run_name: str,
    test_tsv: str,
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
        tokenizer_name=None,  # auto-detect from checkpoint
        beam_size=beam_size,
    )


def eval_with_pi(
    ckpt_path: str,
    run_name: str,
    test_tsv: str,
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

        # Save temp checkpoint with PI applied
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
            tokenizer_name=None,
            beam_size=beam_size,
        )
        result["pi_scale"] = scale
        results.append(result)
        temp_ckpt.unlink(missing_ok=True)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--pi-scales", nargs="+", type=float, default=[1.5, 2.0, 3.0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    ckpt_dir = paths.CHECKPOINT_DIR
    device = args.device

    # Find all checkpoints
    all_ckpts = []
    for subdir in ckpt_dir.iterdir():
        if not subdir.is_dir():
            continue
        best_pt = subdir / "best.pt"
        if best_pt.exists():
            all_ckpts.append((subdir.name, str(best_pt)))

    print(f"[eval] Found {len(all_ckpts)} checkpoints")

    # Language-specific test data
    def get_test_data(run_name: str) -> str | None:
        if "_de" in run_name:
            return "raw_data/wmt14/test.tsv"
        elif "_hi" in run_name:
            return "processed_data/test_3k.tsv"
        elif "_bn" in run_name:
            return "processed_data_bn/test_3k.tsv"
        return None

    for run_name, ckpt_path in sorted(all_ckpts):
        if run_name.startswith("_temp_") or run_name.startswith("_test"):
            continue

        test_tsv = get_test_data(run_name)
        if test_tsv is None:
            print(f"[eval] Skipping {run_name}: unknown language")
            continue
        if not Path(test_tsv).exists():
            print(f"[eval] Skipping {run_name}: test data not found at {test_tsv}")
            continue

        print(f"\n[eval] {'='*60}")
        print(f"[eval] {run_name}")
        print(f"[eval] {'='*60}")

        # Standard eval
        eval_checkpoint(ckpt_path, f"{run_name}_eval", test_tsv, device, args.beam_size)

        # PI eval for standard RoPE checkpoints only (not AdaptiveRoPE or ablations)
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pe_type = str(ckpt_data.get("config", {}).get("pe_type", ""))
        is_rope_only = pe_type == "rope"
        if is_rope_only and "pi" not in run_name and "carope" not in run_name and "ls" not in run_name:
            print(f"[eval] Running PI variants for {run_name}...")
            eval_with_pi(ckpt_path, f"{run_name}_eval", test_tsv, device, args.pi_scales, args.beam_size)

    print(f"\n[eval] All evaluations complete.")


if __name__ == "__main__":
    main()
