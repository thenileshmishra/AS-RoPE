"""Shared Google Drive path configuration for the sequential pipeline.

Every step reads from and writes to paths defined here so that the four
stages can run independently and still find each other's artifacts.

Override `PROJECT_ROOT` with the `NEUR_DRIVE_ROOT` environment variable if
you keep your project under a different Drive folder.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("NEUR_DRIVE_ROOT", "/content/drive/MyDrive/neur"))

RAW_DIR = PROJECT_ROOT / "raw_data"
RAW_SAMANANTAR = RAW_DIR / "samanantar" / "samanantar_hi_en.tsv"
RAW_FLORES = RAW_DIR / "flores200" / "flores200_hi_en_devtest.tsv"

# Bengali-English paths
RAW_SAMANANTAR_BN = RAW_DIR / "samanantar" / "samanantar_bn_en.tsv"
RAW_FLORES_BN = RAW_DIR / "flores200" / "flores200_bn_en_devtest.tsv"
PROCESSED_DIR_BN = PROJECT_ROOT / "processed_data_bn"
PROCESSED_TRAIN_BN = PROCESSED_DIR_BN / "train.tsv"
PROCESSED_VAL_BN = PROCESSED_DIR_BN / "val.tsv"

PROCESSED_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_TRAIN = PROCESSED_DIR / "train.tsv"
PROCESSED_VAL = PROCESSED_DIR / "val.tsv"
PROCESSED_TEST = PROCESSED_DIR / "test.tsv"
PROCESSED_METADATA = PROCESSED_DIR / "metadata.json"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
METRICS_DIR = OUTPUTS_DIR / "metrics"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, CHECKPOINT_DIR, LOGS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    return (
        f"PROJECT_ROOT    = {PROJECT_ROOT}\n"
        f"RAW_SAMANANTAR  = {RAW_SAMANANTAR}\n"
        f"RAW_FLORES      = {RAW_FLORES}\n"
        f"PROCESSED_DIR   = {PROCESSED_DIR}\n"
        f"CHECKPOINT_DIR  = {CHECKPOINT_DIR}\n"
        f"LOGS_DIR        = {LOGS_DIR}\n"
        f"METRICS_DIR     = {METRICS_DIR}"
    )


if __name__ == "__main__":
    print(summary())
