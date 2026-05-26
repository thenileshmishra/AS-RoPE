"""Run evaluation only on checkpoints used in the paper.

Skips legacy inconsistent checkpoints. Uses greedy decoding for speed.
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


# Checkpoints to include in paper results (consistent hyperparameters)
PAPER_CHECKPOINTS = {
    # En-De (seq=128, batch=256, steps=25K, lr=1e-3)
    "rope_de_s43": "raw_data/wmt14/test.tsv",
    "rope_de_s44": "raw_data/wmt14/test.tsv",
    "adaptiverope_de_s43": "raw_data/wmt14/test.tsv",
    "adaptiverope_de_s44": "raw_data/wmt14/test.tsv",
    "gatesonly_de_s42": "raw_data/wmt14/test.tsv",
    "gatesonly_de_s43": "raw_data/wmt14/test.tsv",
    "gatesonly_de_s44": "raw_data/wmt14/test.tsv",
    "phasesonly_de_s42": "raw_data/wmt14/test.tsv",
    "phasesonly_de_s43": "raw_data/wmt14/test.tsv",
    "phasesonly_de_s44": "raw_data/wmt14/test.tsv",
    "sinusoidal_de_correct": "raw_data/wmt14/test.tsv",
    "sinusoidal_de_s43": "raw_data/wmt14/test.tsv",
    "sinusoidal_de_s44": "raw_data/wmt14/test.tsv",
    "alibi_de_s42": "raw_data/wmt14/test.tsv",

    # Hi-En (seq=192, batch=64, steps=75K, lr=5e-4)
    "rope_hi_s42": "processed_data/test_3k.tsv",
    "adaptiverope_hi_s42": "processed_data/test_3k.tsv",
    "gatesonly_hi_s42": "processed_data/test_3k.tsv",
    "phasesonly_hi_s42": "processed_data/test_3k.tsv",

    # Bn-En (seq=192, batch=64, steps=75K, lr=5e-4)
    "rope_bn_s42": "processed_data_bn/test_3k.tsv",
    "adaptiverope_bn_s42": "processed_data_bn/test_3k.tsv",
    "gatesonly_bn_s42": "processed_data_bn/test_3k.tsv",
    "phasesonly_bn_s42": "processed_data_bn/test_3k.tsv",
    "sinusoidal_bn_s42": "processed_data_bn/test_3k.tsv",
}


def eval_with_pi(
    ckpt_path: str,
    run_name: str,
    test_tsv: str,
    device: str,
    scales: list[float],
    beam_size: int = 1,
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
    parser = argparse.ArgumentParser(description="Run paper evaluation")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--pi-scales", nargs="+", type=float, default=[1.5, 2.0, 3.0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    device = args.device
    ckpt_dir = paths.CHECKPOINT_DIR

    for run_name, test_tsv in sorted(PAPER_CHECKPOINTS.items()):
        ckpt_path = ckpt_dir / run_name / "best.pt"
        if not ckpt_path.exists():
            print(f"[eval] WARNING: checkpoint not found: {ckpt_path}")
            continue

        out_dir = paths.METRICS_DIR / f"{run_name}_eval"
        if (out_dir / "eval_summary.json").exists():
            print(f"[eval] {run_name} already evaluated, skipping")
        else:
            print(f"\n[eval] {'='*60}")
            print(f"[eval] {run_name}")
            print(f"[eval] {'='*60}")
            evaluate_checkpoint(str(ckpt_path), test_tsv, str(out_dir), device, beam_size=args.beam_size)

        # PI only for standard RoPE
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pe_type = str(ckpt_data.get("config", {}).get("pe_type", ""))
        if pe_type == "rope":
            print(f"[eval] Running PI variants for {run_name}...")
            eval_with_pi(str(ckpt_path), f"{run_name}_eval", test_tsv, device, args.pi_scales, args.beam_size)

    print("\n[eval] All paper evaluations complete.")


if __name__ == "__main__":
    main()
