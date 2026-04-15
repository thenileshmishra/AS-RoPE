"""Step 1 (Bengali) — Download Samanantar Bn-En + FLORES-200 Bn-En.

Identical to step1_download but for Bengali→English. Writes to separate
output paths (RAW_SAMANANTAR_BN, RAW_FLORES_BN) so it never touches the
existing Hi-En data.

Usage:
    !python -m pipeline.step1_download_bn
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

from datasets import load_dataset

from pipeline import paths


SAMANANTAR_REPO = "ai4bharat/samanantar"
SAMANANTAR_CONFIG = "bn"          # Bengali
FLORES_TAR_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_SPLIT = "devtest"
FLORES_SRC_LANG = "ben_Beng"     # Bengali script
FLORES_TGT_LANG = "eng_Latn"
SEED = 42


def download_samanantar_bn(output_tsv: Path, force: bool = False) -> None:
    if output_tsv.exists() and output_tsv.stat().st_size > 0 and not force:
        print(f"[step1_bn] samanantar-bn already present -> {output_tsv}")
        return

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[step1_bn] loading {SAMANANTAR_REPO}/{SAMANANTAR_CONFIG} ...")
    ds = load_dataset(SAMANANTAR_REPO, SAMANANTAR_CONFIG, split="train")
    print(f"[step1_bn] total rows: {len(ds):,}")

    kept = 0
    with open(output_tsv, "w", encoding="utf-8", buffering=1 << 20) as f:
        for ex in ds:
            tr = ex.get("translation", {})
            bn = (tr.get("bn") or ex.get("src") or "").strip()
            en = (tr.get("en") or ex.get("tgt") or "").strip()
            if not bn or not en:
                continue
            f.write(f"{' '.join(bn.split())}\t{' '.join(en.split())}\n")
            kept += 1
            if kept % 200_000 == 0:
                print(f"[step1_bn] {kept:,} pairs written ...")

    print(f"[step1_bn] wrote {kept:,} pairs -> {output_tsv}")


def download_flores_bn(output_tsv: Path, force: bool = False) -> None:
    if output_tsv.exists() and output_tsv.stat().st_size > 0 and not force:
        print(f"[step1_bn] flores-bn already present -> {output_tsv}")
        return

    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output_tsv.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "flores200_dataset.tar.gz"

    if not tar_path.exists():
        print(f"[step1_bn] downloading flores200 archive ...")
        with urllib.request.urlopen(FLORES_TAR_URL) as resp, open(tar_path, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)

    src_rel = f"flores200_dataset/{FLORES_SPLIT}/{FLORES_SRC_LANG}.{FLORES_SPLIT}"
    tgt_rel = f"flores200_dataset/{FLORES_SPLIT}/{FLORES_TGT_LANG}.{FLORES_SPLIT}"

    with tarfile.open(tar_path, "r:gz") as tar:
        def _resolve(rel):
            try:
                tar.getmember(rel)
                return rel
            except KeyError:
                suffix = f"/{FLORES_SPLIT}/{Path(rel).name}"
                for name in tar.getnames():
                    if name.endswith(suffix):
                        return name
                raise RuntimeError(f"Missing {rel} in tarball")

        src_f = tar.extractfile(_resolve(src_rel))
        tgt_f = tar.extractfile(_resolve(tgt_rel))
        src_lines = src_f.read().decode("utf-8").splitlines()
        tgt_lines = tgt_f.read().decode("utf-8").splitlines()

    kept = 0
    with open(output_tsv, "w", encoding="utf-8") as f:
        for bn, en in zip(src_lines, tgt_lines):
            bn = " ".join((bn or "").split())
            en = " ".join((en or "").split())
            if not bn or not en:
                continue
            f.write(f"{bn}\t{en}\n")
            kept += 1

    print(f"[step1_bn] wrote {kept:,} flores-bn pairs -> {output_tsv}")


def main() -> None:
    paths.ensure_dirs()
    download_samanantar_bn(paths.RAW_SAMANANTAR_BN)
    download_flores_bn(paths.RAW_FLORES_BN)
    print("[step1_bn] done.")


if __name__ == "__main__":
    main()
