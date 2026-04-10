"""Collect Day 5 baseline results into summary CSV and JSON.

Scans each baseline's eval/metrics.json under the output root and
assembles a comparison table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SUMMARY_COLUMNS = [
    "run_name",
    "positional_encoding",
    "bleu",
    "chrf",
    "checkpoint_path",
    "metrics_path",
    "status",
]


def collect(config_path: str | Path) -> list[dict]:
    """Read config, scan each baseline's eval output, return rows."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    rows: list[dict] = []

    for baseline in config["baselines"]:
        name = baseline["name"]
        pe = baseline["positional_encoding"]
        metrics_path = output_root / name / "eval" / "metrics.json"
        ckpt_path = output_root / name / "best.pt"

        row = {
            "run_name": name,
            "positional_encoding": pe,
            "bleu": None,
            "chrf": None,
            "checkpoint_path": str(ckpt_path),
            "metrics_path": str(metrics_path),
            "status": "missing",
        }

        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                row["bleu"] = metrics.get("bleu")
                row["chrf"] = metrics.get("chrf")
                row["status"] = "ok"
            except (json.JSONDecodeError, KeyError) as e:
                row["status"] = f"error: {e}"
        rows.append(row)

    return rows


def write_summary(rows: list[dict], output_root: Path) -> tuple[Path, Path]:
    """Write summary.csv and summary.json to output_root."""
    output_root.mkdir(parents=True, exist_ok=True)

    csv_path = output_root / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_root / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return csv_path, json_path


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Collect Day 5 baseline results.")
    parser.add_argument(
        "--config",
        default="configs/day5_baselines.json",
        help="Path to baseline config JSON",
    )
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])

    rows = collect(args.config)
    csv_path, json_path = write_summary(rows, output_root)

    print(f"[collect] {len(rows)} baselines scanned")
    for row in rows:
        bleu = f"{row['bleu']:.2f}" if row["bleu"] is not None else "N/A"
        chrf = f"{row['chrf']:.2f}" if row["chrf"] is not None else "N/A"
        print(f"  {row['run_name']:20s}  BLEU={bleu:>8s}  chrF={chrf:>8s}  [{row['status']}]")
    print(f"[collect] CSV  -> {csv_path}")
    print(f"[collect] JSON -> {json_path}")

    return {"csv_path": str(csv_path), "json_path": str(json_path), "rows": rows}


if __name__ == "__main__":
    main()
