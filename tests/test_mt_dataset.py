"""Smoke tests for the Hindi-English MT dataset pipeline."""

import tempfile
from pathlib import Path

from data.mt_dataset import (
    MTDataset,
    create_sample_tsv,
    deterministic_split,
    load_pairs,
    load_pairs_jsonl,
    normalize_text,
)
from torch.utils.data import DataLoader


def _make_tsv(tmp: Path, lines: list[str]) -> Path:
    p = tmp / "test.tsv"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _make_jsonl(tmp: Path, objs: list[dict]) -> Path:
    import json
    p = tmp / "test.jsonl"
    p.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in objs), encoding="utf-8")
    return p


def test_normalize_text():
    assert normalize_text("  hello   world  ") == "hello world"
    assert normalize_text("") == ""
    assert normalize_text("हिंदी   वाक्य") == "हिंदी वाक्य"


def test_load_pairs_tsv():
    with tempfile.TemporaryDirectory() as tmp:
        p = _make_tsv(Path(tmp), [
            "नमस्ते\tHello",
            "कैसे हो\tHow are you",
            "bad line without tab",
            "\t",
            "",
        ])
        pairs = load_pairs(p)
        assert len(pairs) == 2
        assert pairs[0] == ("नमस्ते", "Hello")
        assert pairs[1] == ("कैसे हो", "How are you")


def test_load_pairs_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        p = _make_jsonl(Path(tmp), [
            {"src": "नमस्ते", "tgt": "Hello"},
            {"src": "", "tgt": "empty source"},
            {"src": "valid", "tgt": ""},
            {"src": "दो", "tgt": "Two"},
        ])
        pairs = load_pairs_jsonl(p)
        assert len(pairs) == 2
        assert pairs[1] == ("दो", "Two")


def test_deterministic_split_is_stable():
    pairs = [(f"src_{i}", f"tgt_{i}") for i in range(100)]
    s1 = deterministic_split(pairs, seed=7)
    s2 = deterministic_split(pairs, seed=7)
    assert s1 == s2


def test_deterministic_split_covers_all():
    pairs = [(f"src_{i}", f"tgt_{i}") for i in range(200)]
    train, val, test = deterministic_split(pairs)
    assert len(train) + len(val) + len(test) == 200


def test_dataset_raw_strings():
    pairs = [("hello", "world"), ("foo", "bar")]
    ds = MTDataset(pairs)
    assert len(ds) == 2
    assert ds[0] == ("hello", "world")


def test_dataloader_batch():
    """Core smoke test: build a DataLoader and read one full batch."""
    with tempfile.TemporaryDirectory() as tmp:
        tsv_path = create_sample_tsv(Path(tmp) / "sample.tsv", n=30)
        pairs = load_pairs(tsv_path)
        ds = MTDataset(pairs)
        loader = DataLoader(ds, batch_size=4)
        batch = next(iter(loader))
        src_batch, tgt_batch = batch
        assert len(src_batch) == 4
        assert len(tgt_batch) == 4


def test_create_sample_tsv():
    with tempfile.TemporaryDirectory() as tmp:
        p = create_sample_tsv(Path(tmp) / "out.tsv", n=10)
        assert p.exists()
        pairs = load_pairs(p)
        assert len(pairs) == 10


if __name__ == "__main__":
    test_normalize_text()
    test_load_pairs_tsv()
    test_load_pairs_jsonl()
    test_deterministic_split_is_stable()
    test_deterministic_split_covers_all()
    test_dataset_raw_strings()
    test_dataloader_batch()
    test_create_sample_tsv()
    print("All tests passed.")
