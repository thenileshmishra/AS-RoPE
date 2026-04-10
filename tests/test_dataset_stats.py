"""Tests for scripts/dataset_stats.py."""

import json
import tempfile
from pathlib import Path

from scripts.dataset_stats import compute_stats, write_stats


def _make_tsv(tmp: Path, rows: list[str], name: str = "data.tsv") -> Path:
    p = tmp / name
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


def test_compute_stats_basic_counts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = _make_tsv(tmp, [
            "hello world\tनमस्ते दुनिया",
            "good morning\tशुभ प्रभात",
            "thank you very much\tबहुत बहुत धन्यवाद",
        ])
        stats = compute_stats(p)
        assert stats["num_samples"] == 3
        # avg_src tokens: (2+2+4)/3 ≈ 2.67
        assert abs(stats["avg_src_len_tokens"] - (8 / 3)) < 1e-6
        assert stats["empty_or_invalid_rows"] == 0
        assert stats["duplicate_pairs_count"] == 0


def test_compute_stats_invalid_and_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = _make_tsv(tmp, [
            "hello\tworld",
            "no_tab_here",
            "\t",
            "hello\tworld",  # duplicate of first
            "foo\tbar",
        ])
        stats = compute_stats(p)
        # load_pairs filters out invalids and keeps the duplicate
        assert stats["num_samples"] == 3  # 'hello\tworld' x2, 'foo\tbar' x1
        assert stats["empty_or_invalid_rows"] == 2  # no_tab_here, "\t"
        assert stats["duplicate_pairs_count"] == 1


def test_compute_stats_p95():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 20 rows of varying length
        rows = [f"{'x ' * (i + 1)}\t{'y ' * (i + 1)}" for i in range(20)]
        p = _make_tsv(tmp, rows)
        stats = compute_stats(p)
        assert stats["num_samples"] == 20
        assert stats["p95_src_len"] > 0
        assert stats["p95_tgt_len"] > 0
        # p95 should be close to the high end (~19)
        assert stats["p95_src_len"] >= 18.0


def test_write_stats_creates_json():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = _make_tsv(tmp, ["a\tb", "c\td"])
        out = tmp / "out" / "stats.json"
        stats = compute_stats(p)
        write_stats(stats, out)
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["num_samples"] == 2
        assert "avg_src_len_tokens" in loaded
        assert "p95_src_len" in loaded
        assert "duplicate_pairs_count" in loaded


def test_compute_stats_missing_file_raises():
    try:
        compute_stats("/tmp/this_does_not_exist_xyz.tsv")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_compute_stats_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = tmp / "data.jsonl"
        p.write_text("\n".join([
            json.dumps({"src": "hello", "tgt": "नमस्ते"}, ensure_ascii=False),
            json.dumps({"src": "good morning", "tgt": "शुभ प्रभात"}, ensure_ascii=False),
            json.dumps({"src": "", "tgt": "empty"}, ensure_ascii=False),  # invalid
        ]), encoding="utf-8")
        stats = compute_stats(p)
        assert stats["num_samples"] == 2
        assert stats["empty_or_invalid_rows"] == 1


if __name__ == "__main__":
    test_compute_stats_basic_counts()
    test_compute_stats_invalid_and_duplicates()
    test_compute_stats_p95()
    test_write_stats_creates_json()
    test_compute_stats_missing_file_raises()
    test_compute_stats_jsonl()
    print("All dataset_stats tests passed.")
