"""Compute basic dataset stats for MT pre-training sanity checks.

Reads a parallel TSV/JSONL file and writes a JSON of summary stats —
sample counts, token-length distributions, invalid rows, and exact
duplicate pairs. Tokens are counted by whitespace splitting (cheap +
deterministic), suitable for quick pre-cleaning sanity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.mt_dataset import load_pairs


def _percentile(sorted_vals: list[int], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 1]). Empty list -> 0.0."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = q * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo]) * (1 - frac) + float(sorted_vals[hi]) * frac


def _count_invalid_rows(path: Path) -> int:
    """Count rows that fail TSV format (no tab, empty src or empty tgt)."""
    if path.suffix == ".jsonl":
        bad = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    bad += 1
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                if not obj.get("src") or not obj.get("tgt"):
                    bad += 1
        return bad

    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                bad += 1
                continue
            src, _, tgt = line.partition("\t")
            if not src.strip() or not tgt.strip():
                bad += 1
    return bad


def compute_stats(input_path: str | Path) -> dict:
    """Compute dataset stats from a TSV or JSONL file."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    pairs = load_pairs(input_path)
    invalid_rows = _count_invalid_rows(input_path)

    src_lens = [len(s.split()) for s, _ in pairs]
    tgt_lens = [len(t.split()) for _, t in pairs]
    src_sorted = sorted(src_lens)
    tgt_sorted = sorted(tgt_lens)

    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for pair in pairs:
        if pair in seen:
            duplicates += 1
        else:
            seen.add(pair)

    n = len(pairs)
    avg_src = sum(src_lens) / n if n else 0.0
    avg_tgt = sum(tgt_lens) / n if n else 0.0

    return {
        "input_path": str(input_path),
        "num_samples": n,
        "avg_src_len_tokens": avg_src,
        "avg_tgt_len_tokens": avg_tgt,
        "p95_src_len": _percentile(src_sorted, 0.95),
        "p95_tgt_len": _percentile(tgt_sorted, 0.95),
        "empty_or_invalid_rows": invalid_rows,
        "duplicate_pairs_count": duplicates,
    }


def write_stats(stats: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Compute MT dataset stats.")
    parser.add_argument("--input", required=True, help="Path to TSV or JSONL file")
    parser.add_argument(
        "--output",
        default="results/day5_baselines/artifacts/dataset_stats.json",
        help="Output JSON path",
    )
    args = parser.parse_args(argv)

    stats = compute_stats(args.input)
    out = write_stats(stats, args.output)

    print(f"[dataset_stats] input  : {stats['input_path']}")
    print(f"[dataset_stats] samples: {stats['num_samples']}")
    print(f"[dataset_stats] avg src tokens: {stats['avg_src_len_tokens']:.2f}")
    print(f"[dataset_stats] avg tgt tokens: {stats['avg_tgt_len_tokens']:.2f}")
    print(f"[dataset_stats] p95 src tokens: {stats['p95_src_len']:.2f}")
    print(f"[dataset_stats] p95 tgt tokens: {stats['p95_tgt_len']:.2f}")
    print(f"[dataset_stats] invalid rows  : {stats['empty_or_invalid_rows']}")
    print(f"[dataset_stats] duplicates    : {stats['duplicate_pairs_count']}")
    print(f"[dataset_stats] output -> {out}")
    return stats


if __name__ == "__main__":
    main()
