"""Step 1 — Download raw datasets to Google Drive.

Downloads:
    1. Samanantar (Hindi ↔ English), all pairs by default (or capped via --sample-size).
  2. FLORES-200 devtest split for evaluation.

Both datasets are written as raw TSV files (``src\\ttgt`` per line) directly
to Google Drive under :mod:`pipeline.paths`. This step performs NO cleaning
or filtering — that is Step 2's responsibility.

Usage (from the Colab notebook):
    !python -m pipeline.step1_download

The script is idempotent: if a non-empty output file already exists it is
left alone unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

from datasets import load_dataset

from pipeline import paths


SAMANANTAR_REPO = "ai4bharat/samanantar"
SAMANANTAR_CONFIG = "hi"
DEFAULT_SAMPLE_SIZE = 0
SEED = 42

FLORES_TAR_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SPLIT = "devtest"
FLORES_SRC_LANG = "hin_Deva"
FLORES_TGT_LANG = "eng_Latn"


def _devanagari_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for ch in letters if "DEVANAGARI" in unicodedata.name(ch, ""))
    return dev / len(letters)


def _detect_hi_en_keys(example: dict) -> tuple[str, str, bool]:
    """Return (hi_key, en_key, is_translation_dict) for a Samanantar row.

    Samanantar ships in several layouts depending on the config/version:
    ``{"translation": {"hi": ..., "en": ...}}``, ``{"src": ..., "tgt": ...}``,
    or ``{"source": ..., "target": ...}``. This helper picks whichever is
    present and uses Devanagari script ratio as a tiebreaker for which side
    is Hindi.
    """
    if "translation" in example and isinstance(example["translation"], dict):
        return "hi", "en", True
    if "src" in example and "tgt" in example:
        src, tgt = example["src"], example["tgt"]
    elif "source" in example and "target" in example:
        src, tgt = example["source"], example["target"]
    else:
        raise RuntimeError(
            f"Unrecognized Samanantar row schema. Keys: {list(example.keys())}"
        )
    if _devanagari_ratio(str(src)) >= _devanagari_ratio(str(tgt)):
        return "src" if "src" in example else "source", "tgt" if "tgt" in example else "target", False
    return "tgt" if "tgt" in example else "target", "src" if "src" in example else "source", False


def download_samanantar(output_tsv: Path, sample_size: int, force: bool = False) -> Path:
    if output_tsv.exists() and output_tsv.stat().st_size > 0 and not force:
        print(f"[step1] samanantar already present -> {output_tsv} (use --force to redownload)")
        return output_tsv

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[step1] loading {SAMANANTAR_REPO}/{SAMANANTAR_CONFIG} ...")
    ds = load_dataset(SAMANANTAR_REPO, SAMANANTAR_CONFIG, split="train")
    total = len(ds)
    print(f"[step1] samanantar total rows: {total:,}")

    if sample_size and sample_size < total:
        ds = ds.shuffle(seed=SEED).select(range(sample_size))
        print(f"[step1] sampled {sample_size:,} rows (seed={SEED})")
    else:
        print(f"[step1] using all {total:,} rows")

    hi_key, en_key, is_translation_dict = _detect_hi_en_keys(ds[0])
    print(f"[step1] detected hi={hi_key} en={en_key} translation_dict={is_translation_dict}")

    kept = 0
    with open(output_tsv, "w", encoding="utf-8", buffering=1024*1024) as f:
        for i, ex in enumerate(ds):
            if is_translation_dict:
                tr = ex["translation"]
                hi = (tr.get(hi_key) or "").strip()
                en = (tr.get(en_key) or "").strip()
            else:
                hi = (ex.get(hi_key) or "").strip()
                en = (ex.get(en_key) or "").strip()
            if not hi or not en:
                continue
            hi = " ".join(hi.split())
            en = " ".join(en.split())
            f.write(f"{hi}\t{en}\n")
            kept += 1
            if (kept + 1) % 100000 == 0:
                print(f"[step1] progress: {kept:,} pairs written...")

    try:
        size_mb = output_tsv.stat().st_size / 1e6
        print(f"[step1] wrote {kept:,} pairs -> {output_tsv} ({size_mb:.1f} MB)")
    except FileNotFoundError:
        if kept > 0:
            print(f"[step1] wrote {kept:,} pairs -> {output_tsv} (file stat unavailable, likely network delay)")
        else:
            raise RuntimeError(f"Failed to write {output_tsv}: no pairs kept and file not found")
    return output_tsv


def _download_flores_tar(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "flores200_dataset.tar.gz"
    if tar_path.exists() and tar_path.stat().st_size > 0:
        return tar_path
    print(f"[step1] downloading flores200 archive -> {tar_path}")
    with urllib.request.urlopen(FLORES_TAR_URL) as resp, open(tar_path, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return tar_path


def download_flores(output_tsv: Path, force: bool = False) -> Path:
    if output_tsv.exists() and output_tsv.stat().st_size > 0 and not force:
        print(f"[step1] flores already present -> {output_tsv} (use --force to redownload)")
        return output_tsv

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_tsv.parent / ".cache"
    tar_path = _download_flores_tar(cache_dir)

    src_rel = f"flores200_dataset/{FLORES_SPLIT}/{FLORES_SRC_LANG}.{FLORES_SPLIT}"
    tgt_rel = f"flores200_dataset/{FLORES_SPLIT}/{FLORES_TGT_LANG}.{FLORES_SPLIT}"

    def _resolve_member_name(tar: tarfile.TarFile, expected_rel: str) -> str:
        try:
            tar.getmember(expected_rel)
            return expected_rel
        except KeyError:
            suffix = f"/{FLORES_SPLIT}/{Path(expected_rel).name}"
            for name in tar.getnames():
                if name.endswith(suffix):
                    return name
            raise RuntimeError(f"Missing {expected_rel} in tarball")

    print(f"[step1] extracting flores200 ({FLORES_SPLIT}) ...")
    with tarfile.open(tar_path, mode="r:gz") as tar:
        src_member = _resolve_member_name(tar, src_rel)
        tgt_member = _resolve_member_name(tar, tgt_rel)
        src_f = tar.extractfile(src_member)
        tgt_f = tar.extractfile(tgt_member)
        if src_f is None or tgt_f is None:
            raise RuntimeError(f"Missing {src_rel} or {tgt_rel} in tarball")
        src_lines = src_f.read().decode("utf-8").splitlines()
        tgt_lines = tgt_f.read().decode("utf-8").splitlines()

    if len(src_lines) != len(tgt_lines):
        raise RuntimeError(
            f"flores200 line count mismatch: src={len(src_lines)} tgt={len(tgt_lines)}"
        )

    kept = 0
    with open(output_tsv, "w", encoding="utf-8") as f:
        for hi, en in zip(src_lines, tgt_lines):
            hi = " ".join((hi or "").split())
            en = " ".join((en or "").split())
            if not hi or not en:
                continue
            f.write(f"{hi}\t{en}\n")
            kept += 1

    print(f"[step1] wrote {kept:,} flores pairs -> {output_tsv}")
    return output_tsv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Step 1: download raw datasets to Google Drive")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Max Samanantar pairs to download (default: all rows; set >0 to cap)",
    )
    parser.add_argument("--force", action="store_true", help="Redownload even if outputs exist")
    parser.add_argument(
        "--skip-samanantar", action="store_true", help="Skip Samanantar download"
    )
    parser.add_argument("--skip-flores", action="store_true", help="Skip FLORES-200 download")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    print("[step1] paths:")
    print(paths.summary())

    if not args.skip_samanantar:
        download_samanantar(paths.RAW_SAMANANTAR, args.sample_size, force=args.force)
    if not args.skip_flores:
        download_flores(paths.RAW_FLORES, force=args.force)

    print("[step1] done. Proceed to Step 2 (preprocessing).")


if __name__ == "__main__":
    main()
