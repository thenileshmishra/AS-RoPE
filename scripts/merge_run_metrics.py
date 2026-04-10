"""Merge per-baseline training summaries with eval metrics into one table.

For each baseline in the config, reads:
  - <output_root>/<name>/run_summary.json (from training)
  - <output_root>/<name>/eval/metrics.json (from BLEU/chrF eval)

Writes one row per baseline to:
  - <output_root>/tables/runs_merged.csv
  - <output_root>/tables/runs_merged.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MERGED_COLUMNS = [
    "run_name",
    "positional_encoding",
    "bleu",
    "chrf",
    "best_val_loss",
    "best_step",
    "total_train_time_sec",
    "avg_step_time_sec",
    "total_tokens_seen",
    "seed",
    "status",
]


def _safe_load_json(path: Path) -> tuple[dict | None, str | None]:
    """Return (data, error). data is None on failure."""
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as e:
        return None, f"parse_error: {e}"


def merge_one(
    name: str, pe: str, run_dir: Path
) -> dict:
    """Merge one baseline's training + eval results into a single row."""
    summary_path = run_dir / "run_summary.json"
    eval_path = run_dir / "eval" / "metrics.json"

    row = {col: None for col in MERGED_COLUMNS}
    row["run_name"] = name
    row["positional_encoding"] = pe
    statuses: list[str] = []

    summary, err = _safe_load_json(summary_path)
    if summary is not None:
        row["best_val_loss"] = summary.get("best_val_loss")
        row["best_step"] = summary.get("best_step")
        row["total_train_time_sec"] = summary.get("total_train_time_sec")
        row["avg_step_time_sec"] = summary.get("avg_step_time_sec")
        row["total_tokens_seen"] = summary.get("total_tokens_seen")
        row["seed"] = summary.get("seed")
    else:
        statuses.append(f"train_{err}")

    eval_data, err = _safe_load_json(eval_path)
    if eval_data is not None:
        row["bleu"] = eval_data.get("bleu")
        row["chrf"] = eval_data.get("chrf")
    else:
        statuses.append(f"eval_{err}")

    if not statuses:
        row["status"] = "ok"
    else:
        row["status"] = "; ".join(statuses)

    return row


def merge(config_path: str | Path) -> list[dict]:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])

    rows: list[dict] = []
    for baseline in config["baselines"]:
        run_dir = output_root / baseline["name"]
        rows.append(merge_one(baseline["name"], baseline["positional_encoding"], run_dir))
    return rows


def write_merged(rows: list[dict], output_root: Path) -> tuple[Path, Path]:
    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "runs_merged.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MERGED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = tables_dir / "runs_merged.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Merge train + eval metrics per baseline.")
    parser.add_argument("--config", default="configs/day5_baselines.json")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])

    rows = merge(args.config)
    csv_path, json_path = write_merged(rows, output_root)

    print(f"[merge] {len(rows)} baselines merged")
    for row in rows:
        bleu = f"{row['bleu']:.2f}" if isinstance(row["bleu"], (int, float)) else "N/A"
        chrf = f"{row['chrf']:.2f}" if isinstance(row["chrf"], (int, float)) else "N/A"
        print(f"  {row['run_name']:20s}  BLEU={bleu:>8s}  chrF={chrf:>8s}  [{row['status']}]")
    print(f"[merge] CSV  -> {csv_path}")
    print(f"[merge] JSON -> {json_path}")
    return {"csv_path": str(csv_path), "json_path": str(json_path), "rows": rows}


if __name__ == "__main__":
    main()
