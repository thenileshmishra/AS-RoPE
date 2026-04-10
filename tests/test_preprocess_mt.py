"""Tests for the MT preprocessing and caching pipeline."""

import json
import tempfile
from pathlib import Path

import torch

from data.preprocess_mt import (
    deduplicate_pairs,
    preprocess,
    read_tsv_pairs,
    write_tsv,
)


def _make_raw_tsv(tmp: Path, n: int = 30, dupes: int = 0) -> Path:
    """Create a raw TSV with n unique pairs + dupes exact duplicates."""
    p = tmp / "raw.tsv"
    lines: list[str] = []
    for i in range(n):
        lines.append(f"स्रोत_{i}\ttarget_{i}")
    for i in range(dupes):
        lines.append(f"स्रोत_0\ttarget_0")  # duplicate of first pair
    # add some bad lines for cleaning
    lines.append("")
    lines.append("no tab here")
    lines.append("\t")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _make_raw_jsonl(tmp: Path, n: int = 20) -> Path:
    """Create a raw JSONL with n pairs."""
    p = tmp / "raw.jsonl"
    objs = [json.dumps({"src": f"स्रोत_{i}", "tgt": f"target_{i}"}, ensure_ascii=False) for i in range(n)]
    p.write_text("\n".join(objs), encoding="utf-8")
    return p


# ── unit tests ──────────────────────────────────────────────────────

def test_deduplicate_pairs():
    pairs = [("a", "b"), ("c", "d"), ("a", "b"), ("e", "f"), ("c", "d")]
    result = deduplicate_pairs(pairs)
    assert result == [("a", "b"), ("c", "d"), ("e", "f")]


def test_write_and_read_tsv():
    with tempfile.TemporaryDirectory() as tmp:
        pairs = [("hello", "world"), ("foo", "bar")]
        p = Path(tmp) / "out.tsv"
        write_tsv(pairs, p)
        assert p.exists()
        loaded = read_tsv_pairs(p)
        assert loaded == pairs


# ── integration: preprocess pipeline ────────────────────────────────

def test_preprocess_creates_outputs():
    """All expected output files are created."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = _make_raw_tsv(tmp, n=30)
        out = tmp / "processed"

        meta = preprocess(raw, out, seed=42)

        assert (out / "train.tsv").exists()
        assert (out / "val.tsv").exists()
        assert (out / "test.tsv").exists()
        assert (out / "metadata.json").exists()


def test_preprocess_metadata_correctness():
    """Metadata file reflects actual counts."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = _make_raw_tsv(tmp, n=25, dupes=5)
        out = tmp / "processed"

        meta = preprocess(raw, out, seed=42, dedupe=True)

        assert meta["loaded_pairs"] == 30  # 25 unique + 5 dupes
        assert meta["cleaned_pairs"] == 25  # after dedup
        assert meta["train_size"] + meta["val_size"] + meta["test_size"] == 25
        assert meta["seed"] == 42
        assert meta["dedupe"] is True

        # Verify the saved JSON matches
        saved = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert saved["cleaned_pairs"] == meta["cleaned_pairs"]
        assert saved["train_size"] == meta["train_size"]


def test_deterministic_split_stability():
    """Same seed produces identical splits across runs."""
    with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
        tmp1, tmp2 = Path(tmp1), Path(tmp2)
        # Create identical raw files
        raw1 = _make_raw_tsv(tmp1, n=50)
        raw2 = tmp2 / "raw.tsv"
        raw2.write_text(raw1.read_text(encoding="utf-8"), encoding="utf-8")

        out1 = tmp1 / "proc"
        out2 = tmp2 / "proc"

        preprocess(raw1, out1, seed=99)
        preprocess(raw2, out2, seed=99)

        for split in ("train.tsv", "val.tsv", "test.tsv"):
            content1 = (out1 / split).read_text(encoding="utf-8")
            content2 = (out2 / split).read_text(encoding="utf-8")
            assert content1 == content2, f"{split} differs between runs"


def test_different_seeds_give_different_splits():
    """Different seeds produce different train contents."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = _make_raw_tsv(tmp, n=100)

        out1 = tmp / "proc1"
        out2 = tmp / "proc2"

        preprocess(raw, out1, seed=1)
        preprocess(raw, out2, seed=2)

        train1 = (out1 / "train.tsv").read_text(encoding="utf-8")
        train2 = (out2 / "train.tsv").read_text(encoding="utf-8")
        assert train1 != train2


def test_preprocess_no_dedupe():
    """Dedup off preserves duplicates."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = _make_raw_tsv(tmp, n=10, dupes=5)
        out = tmp / "proc"

        meta = preprocess(raw, out, dedupe=False)
        assert meta["loaded_pairs"] == 15
        assert meta["cleaned_pairs"] == 15  # no dedup, so same


def test_preprocess_jsonl_input():
    """JSONL format is handled correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = _make_raw_jsonl(tmp, n=20)
        out = tmp / "proc"

        meta = preprocess(raw, out)
        assert meta["loaded_pairs"] == 20
        assert meta["train_size"] + meta["val_size"] + meta["test_size"] == 20


def test_tokenized_cache_generation():
    """Tokenized .pt caches are generated and loadable."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("SKIP: transformers not installed")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = _make_raw_tsv(tmp, n=10)
        out = tmp / "proc"

        meta = preprocess(
            raw, out, seed=42, max_length=32, tokenizer_name="gpt2",
        )

        # Cache files exist
        for split in ("train", "val", "test"):
            pt_path = out / f"{split}.pt"
            assert pt_path.exists(), f"{split}.pt not found"
            data = torch.load(pt_path, weights_only=True)
            assert "src_ids" in data
            assert "tgt_ids" in data
            assert data["src_ids"].dtype == torch.long
            assert data["src_ids"].shape[1] == 32  # max_length

        # Tokenizer config saved
        assert (out / "tokenizer_config.json").exists()
        tok_cfg = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
        assert tok_cfg["tokenizer_name"] == "gpt2"
        assert tok_cfg["max_length"] == 32

        # Metadata references cache paths
        assert len(meta["cache_paths"]) == 3


def test_preprocess_missing_file_error():
    """Clear error when raw file does not exist."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            preprocess(Path(tmp) / "nonexistent.tsv", Path(tmp) / "out")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "nonexistent.tsv" in str(e)


def test_split_files_content_matches_metadata():
    """The actual pair counts in split files match metadata."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = _make_raw_tsv(tmp, n=40)
        out = tmp / "proc"

        meta = preprocess(raw, out)

        train_pairs = read_tsv_pairs(out / "train.tsv")
        val_pairs = read_tsv_pairs(out / "val.tsv")
        test_pairs = read_tsv_pairs(out / "test.tsv")

        assert len(train_pairs) == meta["train_size"]
        assert len(val_pairs) == meta["val_size"]
        assert len(test_pairs) == meta["test_size"]


if __name__ == "__main__":
    test_deduplicate_pairs()
    test_write_and_read_tsv()
    test_preprocess_creates_outputs()
    test_preprocess_metadata_correctness()
    test_deterministic_split_stability()
    test_different_seeds_give_different_splits()
    test_preprocess_no_dedupe()
    test_preprocess_jsonl_input()
    test_tokenized_cache_generation()
    test_preprocess_missing_file_error()
    test_split_files_content_matches_metadata()
    print("All preprocessing tests passed.")
