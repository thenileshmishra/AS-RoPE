"""Shared path configuration for the WMT14 En-De pipeline."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("NEUR_DRIVE_ROOT", str(REPO_ROOT)))

RAW_DIR = PROJECT_ROOT / "raw_data"
WMT14_DIR = RAW_DIR / "wmt14"
RAW_WMT14_TRAIN = WMT14_DIR / "train.tsv"
RAW_WMT14_VAL = WMT14_DIR / "val.tsv"
RAW_WMT14_TEST = WMT14_DIR / "test.tsv"

PROCESSED_DIR = PROJECT_ROOT / "processed_data_wmt14"
TOKENIZED_DIR = PROCESSED_DIR / "tokenized"
TOKENIZED_TRAIN = TOKENIZED_DIR / "train_wmt14_en_de.pt"
TOKENIZED_VAL = TOKENIZED_DIR / "val_wmt14_en_de.pt"
TOKENIZED_TEST = TOKENIZED_DIR / "test_wmt14_en_de.pt"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
METRICS_DIR = OUTPUTS_DIR / "metrics"


def ensure_dirs() -> None:
    for d in (RAW_DIR, WMT14_DIR, PROCESSED_DIR, TOKENIZED_DIR,
              OUTPUTS_DIR, CHECKPOINT_DIR, LOGS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    return (
        f"PROJECT_ROOT    = {PROJECT_ROOT}\n"
        f"RAW_WMT14_TRAIN = {RAW_WMT14_TRAIN}\n"
        f"RAW_WMT14_VAL   = {RAW_WMT14_VAL}\n"
        f"RAW_WMT14_TEST  = {RAW_WMT14_TEST}\n"
        f"TOKENIZED_DIR   = {TOKENIZED_DIR}\n"
        f"CHECKPOINT_DIR  = {CHECKPOINT_DIR}\n"
        f"LOGS_DIR        = {LOGS_DIR}\n"
        f"METRICS_DIR     = {METRICS_DIR}"
    )


if __name__ == "__main__":
    print(summary())
