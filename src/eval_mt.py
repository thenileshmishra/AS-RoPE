"""Day 4: BLEU + chrF evaluation for MT predictions.

Reads aligned prediction and reference line files, computes corpus-level
BLEU and chrF using sacrebleu, and writes metrics.json. Also supports an
end-to-end mode that invokes src.decode_mt first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ── file helpers ────────────────────────────────────────────────

def read_lines(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# ── metric computation ─────────────────────────────────────────

def compute_bleu_chrf(preds: list[str], refs: list[str]) -> dict:
    """Compute corpus BLEU + chrF via sacrebleu.

    Fails with a clear message if sacrebleu is not installed.
    """
    if len(preds) != len(refs):
        raise ValueError(
            f"Prediction/reference count mismatch: preds={len(preds)}, refs={len(refs)}"
        )
    if len(preds) == 0:
        raise ValueError("Empty prediction/reference lists")

    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError:
        raise RuntimeError(
            "sacrebleu is required for MT evaluation. "
            "Install with: pip install sacrebleu"
        )

    bleu_metric = BLEU()
    chrf_metric = CHRF()
    bleu_score = bleu_metric.corpus_score(preds, [refs])
    chrf_score = chrf_metric.corpus_score(preds, [refs])

    return {
        "bleu": float(bleu_score.score),
        "chrf": float(chrf_score.score),
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
        "num_samples": len(preds),
    }


def evaluate_files(
    pred_file: str | Path,
    ref_file: str | Path,
    output_json: str | Path,
) -> dict:
    """Read pred/ref files, compute metrics, write JSON. Returns the metrics."""
    preds = read_lines(pred_file)
    refs = read_lines(ref_file)
    metrics = compute_bleu_chrf(preds, refs)
    metrics["pred_file"] = str(pred_file)
    metrics["ref_file"] = str(ref_file)

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_sample_outputs(
    src_texts: list[str],
    preds: list[str],
    refs: list[str],
    path: Path,
) -> None:
    """Write src/pred/ref triplets as JSONL for qualitative inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s, p, r in zip(src_texts, preds, refs):
            f.write(json.dumps({"src": s, "pred": p, "ref": r}, ensure_ascii=False) + "\n")


# ── end-to-end runner ──────────────────────────────────────────

def run_end_to_end(
    checkpoint: str | Path,
    eval_tsv: str | Path,
    output_dir: str | Path,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Decode + evaluate in one call. Produces pred.txt, ref.txt, metrics.json, sample_outputs.jsonl."""
    from data.mt_dataset import load_pairs
    from src.decode_mt import run_decode

    output_dir = Path(output_dir)
    pred_file = output_dir / "pred.txt"
    ref_file = output_dir / "ref.txt"
    metrics_file = output_dir / "metrics.json"
    samples_file = output_dir / "sample_outputs.jsonl"

    run_decode(
        checkpoint=checkpoint,
        input_tsv=eval_tsv,
        output_pred=pred_file,
        output_ref=ref_file,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    metrics = evaluate_files(pred_file, ref_file, metrics_file)

    pairs = load_pairs(eval_tsv)
    src_texts = [s for s, _ in pairs]
    preds = read_lines(pred_file)
    refs = read_lines(ref_file)
    write_sample_outputs(src_texts, preds, refs, samples_file)

    metrics["sample_outputs"] = str(samples_file)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# ── CLI ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute BLEU + chrF for MT predictions.")
    sub = p.add_subparsers(dest="mode")

    # Default mode: evaluate existing files
    p.add_argument("--pred-file", help="Prediction text file (one per line)")
    p.add_argument("--ref-file", help="Reference text file (one per line)")
    p.add_argument("--output-json", help="Output metrics.json path")

    # End-to-end mode: decode + evaluate
    e2e = sub.add_parser("e2e", help="Decode from checkpoint and evaluate in one call")
    e2e.add_argument("--checkpoint", required=True)
    e2e.add_argument("--eval-tsv", required=True)
    e2e.add_argument("--output-dir", required=True)
    e2e.add_argument("--max-new-tokens", type=int, default=64)
    e2e.add_argument("--device", default="cpu")

    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)

    if args.mode == "e2e":
        print(f"[eval_mt] e2e: checkpoint={args.checkpoint} eval_tsv={args.eval_tsv}")
        metrics = run_end_to_end(
            checkpoint=args.checkpoint,
            eval_tsv=args.eval_tsv,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    else:
        if not (args.pred_file and args.ref_file and args.output_json):
            raise SystemExit("--pred-file, --ref-file, and --output-json are required")
        metrics = evaluate_files(args.pred_file, args.ref_file, args.output_json)

    print(f"[eval_mt] BLEU = {metrics['bleu']:.4f}")
    print(f"[eval_mt] chrF = {metrics['chrf']:.4f}")
    print(f"[eval_mt] samples = {metrics['num_samples']}")
    print(f"[eval_mt] signatures: bleu={metrics['bleu_signature']}")
    print(f"[eval_mt]             chrf={metrics['chrf_signature']}")
    return metrics


if __name__ == "__main__":
    main()
