import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {"method", "length", "perplexity"}


def _to_rows_from_record(record: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if REQUIRED_COLUMNS.issubset(record.keys()):
        rows.append(
            {
                "method": str(record["method"]),
                "length": int(record["length"]),
                "perplexity": float(record["perplexity"]),
                "source_file": str(source),
                "seed": record.get("seed"),
            }
        )
        return rows

    method = record.get("method") or record.get("variant") or record.get("positional_encoding")
    seed = record.get("seed")

    perplexity_map = (
        record.get("perplexity_by_length")
        or record.get("perplexity")
        or record.get("ppl_by_context")
        or record.get("context_perplexity")
    )

    if method is not None and isinstance(perplexity_map, dict):
        for length, ppl in perplexity_map.items():
            rows.append(
                {
                    "method": str(method),
                    "length": int(length),
                    "perplexity": float(ppl),
                    "source_file": str(source),
                    "seed": seed,
                }
            )
        return rows

    if method is not None and isinstance(record.get("results"), list):
        for item in record["results"]:
            if not isinstance(item, dict):
                continue
            if "length" in item and ("perplexity" in item or "ppl" in item):
                rows.append(
                    {
                        "method": str(method),
                        "length": int(item["length"]),
                        "perplexity": float(item.get("perplexity", item.get("ppl"))),
                        "source_file": str(source),
                        "seed": item.get("seed", seed),
                    }
                )
        return rows

    return rows


def load_json_rows(paths: list[Path]) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    all_rows.extend(_to_rows_from_record(item, path))
        elif isinstance(payload, dict):
            all_rows.extend(_to_rows_from_record(payload, path))

    if not all_rows:
        raise ValueError(
            "No valid perplexity rows found. Expected fields like "
            "{method,length,perplexity} or {method, perplexity_by_length}."
        )

    df = pd.DataFrame(all_rows)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after parsing: {missing}")

    return df


def aggregate_perplexity(df: pd.DataFrame, rope_method: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["method", "length"], as_index=False)["perplexity"]
        .agg(mean="mean", std="std")
        .fillna({"std": 0.0})
    )

    rope_ref = grouped[grouped["method"].str.lower() == rope_method.lower()][["length", "mean"]].rename(
        columns={"mean": "rope_mean"}
    )

    merged = grouped.merge(rope_ref, on="length", how="left")
    merged["pct_improvement_vs_rope"] = (
        (merged["rope_mean"] - merged["mean"]) / merged["rope_mean"] * 100.0
    )
    merged.loc[merged["method"].str.lower() == rope_method.lower(), "pct_improvement_vs_rope"] = 0.0

    merged["Mean"] = merged["mean"].map(lambda x: f"{x:.6f}")
    merged["Std"] = merged["std"].map(lambda x: f"{x:.6f}")
    merged["Mean ± Std"] = merged.apply(lambda r: f"{r['mean']:.6f} ± {r['std']:.6f}", axis=1)
    merged["% Improvement vs RoPE"] = merged["pct_improvement_vs_rope"].map(
        lambda x: "NA" if pd.isna(x) else f"{x:.2f}"
    )

    merged = merged.rename(columns={"method": "Method", "length": "Length"})

    rope_first = merged["Method"].str.lower() == rope_method.lower()
    merged = merged.assign(_rope_first=rope_first)
    merged = merged.sort_values(by=["Length", "_rope_first", "Method"], ascending=[True, False, True]).drop(
        columns=["_rope_first", "mean", "std", "rope_mean", "pct_improvement_vs_rope"]
    )

    return merged[["Method", "Length", "Mean", "Std", "% Improvement vs RoPE", "Mean ± Std"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate perplexity JSON results across seeds/methods.")
    parser.add_argument(
        "--input_glob",
        type=str,
        required=True,
        help="Glob for JSON result files, e.g. 'results/train_runs/**/*.json'",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/perplexity_aggregate.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--rope_method",
        type=str,
        default="rope",
        help="Method name to use as RoPE baseline (default: rope)",
    )
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(f"No JSON files found for glob: {args.input_glob}")

    df = load_json_rows(paths)
    table = aggregate_perplexity(df, rope_method=args.rope_method)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    print("Mean ± std table")
    print(table.to_string(index=False))
    print(f"\nSaved CSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
