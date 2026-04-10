"""Day 3 smoke test: tiny end-to-end MT training run on CPU."""

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch

from src.train_mt import (
    build_mt_example,
    collate_mt_batch,
    train,
)


# ── helpers ─────────────────────────────────────────────────────

def _write_unique_tsv(path: Path, n: int) -> None:
    """Write n unique Hindi-English pairs (so dedup can't collapse them)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            hi = f"यह वाक्य संख्या {i} है।"
            en = f"This is sentence number {i}."
            f.write(f"{hi}\t{en}\n")


class _StubTokenizer:
    """Minimal tokenizer-like object for unit tests that don't need HF."""

    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 99
        self.vocab_size = 100

    def __call__(self, text, add_special_tokens=False):
        # Map each character to (ord(c) % 90) + 1 so we stay away from eos/pad.
        ids = [((ord(c) % 90) + 1) for c in text]
        return {"input_ids": ids}


# ── unit tests: sequence construction & masking ────────────────

def test_build_mt_example_masks_source():
    tok = _StubTokenizer()
    ex = build_mt_example("abc", "xy", tok, max_seq_len=32)
    # src_ids has len 3, tgt_ids has len 2
    # full = [s0,s1,s2, SEP, t0,t1, EOS]  (len 7)
    # input = full[:-1] (len 6), labels = full[1:] (len 6)
    assert len(ex["input_ids"]) == 6
    assert len(ex["labels"]) == 6
    # First 3 label positions must be masked (source segment)
    assert ex["labels"][:3] == [-100, -100, -100]
    # Remaining positions should be real token ids (target + EOS)
    assert all(lbl != -100 for lbl in ex["labels"][3:])
    assert ex["src_len"] == 3
    assert ex["tgt_len"] == 2


def test_build_mt_example_respects_max_seq_len():
    tok = _StubTokenizer()
    ex = build_mt_example("a" * 100, "b" * 100, tok, max_seq_len=16)
    # full length <= max_seq_len, so input_ids length <= max_seq_len - 1
    assert len(ex["input_ids"]) <= 16 - 1
    assert len(ex["labels"]) == len(ex["input_ids"])


def test_collate_mt_batch_pads_and_masks():
    tok = _StubTokenizer()
    ex1 = build_mt_example("abc", "xy", tok, max_seq_len=32)
    ex2 = build_mt_example("a", "xyz", tok, max_seq_len=32)
    batch = collate_mt_batch([ex1, ex2], pad_id=99)

    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"].shape == batch["input_ids"].shape

    # Padding positions must have label -100
    for i in range(batch["input_ids"].shape[0]):
        pad_positions = batch["attention_mask"][i] == 0
        assert (batch["labels"][i][pad_positions] == -100).all()


# ── smoke test: end-to-end training ─────────────────────────────

def test_train_mt_smoke_end_to_end():
    """5-20 training steps on tiny synthetic data; verify artifacts + finite loss."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Create train.tsv and val.tsv with enough distinct pairs for both splits
        train_path = tmp / "train.tsv"
        val_path = tmp / "val.tsv"
        _write_unique_tsv(train_path, n=60)
        _write_unique_tsv(val_path, n=8)

        save_dir = tmp / "run"

        args = argparse.Namespace(
            train_file=str(train_path),
            val_file=str(val_path),
            tokenizer="gpt2",
            positional_encoding="rope",
            max_seq_len=32,
            batch_size=4,
            learning_rate=3e-4,
            weight_decay=0.01,
            num_steps=10,
            eval_every=5,
            save_dir=str(save_dir),
            seed=42,
            device="cpu",
            d_model=64,
            n_layers=2,
            n_heads=2,
        )

        result = train(args)

        # Artifact files exist
        assert Path(result["best_ckpt"]).exists(), "best checkpoint missing"
        assert Path(result["last_ckpt"]).exists(), "last checkpoint missing"
        assert Path(result["metrics_path"]).exists(), "metrics file missing"
        assert Path(result["run_config_path"]).exists(), "run_config.json missing"

        # Metrics JSONL: every line is valid JSON with finite train_loss
        lines = Path(result["metrics_path"]).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == args.num_steps
        for line in lines:
            row = json.loads(line)
            assert "step" in row
            assert "train_loss" in row
            assert math.isfinite(row["train_loss"]), f"non-finite train_loss: {row}"
            assert "lr" in row

        # At least one row has val_loss and it is finite
        val_rows = [json.loads(line) for line in lines if "val_loss" in json.loads(line)]
        assert val_rows, "no validation row recorded"
        assert all(math.isfinite(r["val_loss"]) for r in val_rows), "non-finite val_loss"

        # Best checkpoint loads and contains config
        ckpt = torch.load(result["best_ckpt"], map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        assert "config" in ckpt
        assert ckpt["config"]["positional_encoding"] == "rope"


def test_train_mt_sinusoidal_smoke():
    """Short sinusoidal training run: verify checkpoint + finite loss."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        train_path = tmp / "train.tsv"
        val_path = tmp / "val.tsv"
        _write_unique_tsv(train_path, n=40)
        _write_unique_tsv(val_path, n=8)

        save_dir = tmp / "run_sin"
        args = argparse.Namespace(
            train_file=str(train_path),
            val_file=str(val_path),
            tokenizer="gpt2",
            positional_encoding="sinusoidal",
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

        assert Path(result["best_ckpt"]).exists()
        assert Path(result["metrics_path"]).exists()

        lines = Path(result["metrics_path"]).read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            row = json.loads(line)
            assert math.isfinite(row["train_loss"])

        ckpt = torch.load(result["best_ckpt"], map_location="cpu", weights_only=False)
        assert ckpt["config"]["positional_encoding"] == "sinusoidal"


if __name__ == "__main__":
    test_build_mt_example_masks_source()
    test_build_mt_example_respects_max_seq_len()
    test_collate_mt_batch_pads_and_masks()
    test_train_mt_smoke_end_to_end()
    test_train_mt_sinusoidal_smoke()
    print("All train_mt smoke tests passed.")
