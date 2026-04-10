"""Day 4 smoke test: train tiny checkpoint -> decode -> evaluate end-to-end."""

import argparse
import json
import math
import tempfile
from pathlib import Path

from src.decode_mt import run_decode
from src.eval_mt import compute_bleu_chrf, evaluate_files, run_end_to_end
from src.train_mt import train


def _write_unique_tsv(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            hi = f"यह वाक्य संख्या {i} है।"
            en = f"This is sentence number {i}."
            f.write(f"{hi}\t{en}\n")


def _train_tiny_checkpoint(tmp: Path) -> tuple[Path, Path]:
    """Train a 5-step tiny model. Returns (best_ckpt_path, val_tsv_path)."""
    train_tsv = tmp / "train.tsv"
    val_tsv = tmp / "val.tsv"
    _write_unique_tsv(train_tsv, n=60)
    _write_unique_tsv(val_tsv, n=8)

    save_dir = tmp / "run"
    args = argparse.Namespace(
        train_file=str(train_tsv),
        val_file=str(val_tsv),
        tokenizer="gpt2",
        positional_encoding="rope",
        max_seq_len=32,
        batch_size=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        num_steps=5,
        eval_every=5,
        save_dir=str(save_dir),
        seed=42,
        device="cpu",
        d_model=64,
        n_layers=2,
        n_heads=2,
    )
    result = train(args)
    return Path(result["best_ckpt"]), val_tsv


# ── unit tests for metric computation ─────────────────────────

def test_compute_bleu_chrf_perfect():
    # Long enough to contain 4-grams for corpus BLEU
    preds = [
        "the quick brown fox jumps over the lazy dog",
        "machine translation evaluation is working correctly today",
    ]
    refs = list(preds)
    metrics = compute_bleu_chrf(preds, refs)
    assert metrics["bleu"] > 99.0  # perfect 4-gram match
    assert metrics["chrf"] > 99.0
    assert metrics["num_samples"] == 2
    assert math.isfinite(metrics["bleu"])
    assert math.isfinite(metrics["chrf"])


def test_compute_bleu_chrf_mismatch():
    preds = ["hello"]
    refs = ["goodbye"]
    metrics = compute_bleu_chrf(preds, refs)
    assert metrics["bleu"] >= 0.0
    assert metrics["chrf"] >= 0.0
    assert math.isfinite(metrics["bleu"])
    assert math.isfinite(metrics["chrf"])


def test_evaluate_files_writes_json():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pred = tmp / "pred.txt"
        ref = tmp / "ref.txt"
        out = tmp / "metrics.json"
        pred.write_text("hello\nworld\n", encoding="utf-8")
        ref.write_text("hello\nworld\n", encoding="utf-8")

        metrics = evaluate_files(pred, ref, out)
        assert out.exists()
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["num_samples"] == 2
        assert "bleu" in saved
        assert "chrf" in saved


def test_compute_bleu_chrf_length_mismatch_raises():
    try:
        compute_bleu_chrf(["a", "b"], ["a"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "mismatch" in str(e).lower()


# ── end-to-end decode + evaluate ───────────────────────────────

def test_decode_writes_aligned_pred_ref():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt, val_tsv = _train_tiny_checkpoint(tmp)

        out = tmp / "eval"
        result = run_decode(
            checkpoint=ckpt,
            input_tsv=val_tsv,
            output_pred=out / "pred.txt",
            output_ref=out / "ref.txt",
            max_new_tokens=8,
            device="cpu",
        )

        pred_lines = (out / "pred.txt").read_text(encoding="utf-8").splitlines()
        ref_lines = (out / "ref.txt").read_text(encoding="utf-8").splitlines()

        assert result["num_pairs"] == 8
        assert len(pred_lines) == 8
        assert len(ref_lines) == 8
        assert len(pred_lines) == len(ref_lines)
        # References must match the val.tsv target side
        assert ref_lines[0] == "This is sentence number 0."


def test_end_to_end_produces_all_artifacts():
    """Full Day 4 pipeline: train -> decode -> evaluate, verify metrics.json."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ckpt, val_tsv = _train_tiny_checkpoint(tmp)

        eval_dir = tmp / "eval"
        metrics = run_end_to_end(
            checkpoint=ckpt,
            eval_tsv=val_tsv,
            output_dir=eval_dir,
            max_new_tokens=8,
            device="cpu",
        )

        # Expected files
        assert (eval_dir / "pred.txt").exists()
        assert (eval_dir / "ref.txt").exists()
        assert (eval_dir / "metrics.json").exists()
        assert (eval_dir / "sample_outputs.jsonl").exists()

        # metrics.json is numeric and finite
        saved = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
        assert isinstance(saved["bleu"], (int, float))
        assert isinstance(saved["chrf"], (int, float))
        assert math.isfinite(saved["bleu"])
        assert math.isfinite(saved["chrf"])
        assert saved["num_samples"] == 8

        # sample_outputs.jsonl has aligned src/pred/ref triplets
        lines = (eval_dir / "sample_outputs.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 8
        first = json.loads(lines[0])
        assert "src" in first and "pred" in first and "ref" in first


if __name__ == "__main__":
    test_compute_bleu_chrf_perfect()
    test_compute_bleu_chrf_mismatch()
    test_evaluate_files_writes_json()
    test_compute_bleu_chrf_length_mismatch_raises()
    test_decode_writes_aligned_pred_ref()
    test_end_to_end_produces_all_artifacts()
    print("All Day 4 eval_mt smoke tests passed.")
