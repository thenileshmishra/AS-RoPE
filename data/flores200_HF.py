import argparse
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path


TAR_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SRC_LANG = "hin_Deva"
TGT_LANG = "eng_Latn"
DEFAULT_SPLIT = "devtest"
RAW_OUT = "datasets/eval/flores200_hi_en_devtest.tsv"
CACHE_DIR = Path(".cache/flores200_dataset")


def norm(text: str) -> str:
    return " ".join((text or "").strip().split())


def download_tarball(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / "flores200_dataset.tar.gz"
    if tar_path.exists() and tar_path.stat().st_size > 0:
        return tar_path

    print(f"Downloading FLORES-200 archive -> {tar_path}")
    with urllib.request.urlopen(TAR_URL) as response, open(tar_path, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    return tar_path


def _safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target_dir: Path) -> Path:
    member_path = Path(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise RuntimeError(f"Unsafe member path in tarball: {member.name}")
    try:
        tar.extract(member, path=target_dir, filter="data")
    except TypeError:
        tar.extract(member, path=target_dir)
    return target_dir / member.name.lstrip("./")


def extract_required_files(tar_path: Path, cache_dir: Path, split: str) -> Path:
    root_dir = cache_dir / "flores200_dataset"
    src_file = root_dir / split / f"{SRC_LANG}.{split}"
    tgt_file = root_dir / split / f"{TGT_LANG}.{split}"
    metadata_file = root_dir / f"metadata_{split}.tsv"

    if src_file.exists() and tgt_file.exists() and metadata_file.exists():
        return root_dir

    print(f"Extracting FLORES-200 archive -> {cache_dir}")
    with tarfile.open(tar_path, mode="r:gz") as tar:
        needed = []
        wanted_suffixes = {
            f"flores200_dataset/{split}/{SRC_LANG}.{split}",
            f"flores200_dataset/{split}/{TGT_LANG}.{split}",
            f"flores200_dataset/metadata_{split}.tsv",
        }
        for member in tar.getmembers():
            normalized = member.name.lstrip("./")
            if normalized in wanted_suffixes:
                needed.append(member)

        if not needed:
            raise RuntimeError(
                f"Could not find required FLORES-200 files for split='{split}' in tarball"
            )

        for member in needed:
            _safe_extract_member(tar, member, cache_dir)

    return root_dir


def build_hi_en_tsv(split: str, output_path: str | Path, cache_dir: Path, limit: int | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tar_path = download_tarball(cache_dir)
    root_dir = extract_required_files(tar_path, cache_dir, split)

    src_path = root_dir / split / f"{SRC_LANG}.{split}"
    tgt_path = root_dir / split / f"{TGT_LANG}.{split}"
    metadata_path = root_dir / f"metadata_{split}.tsv"

    print(f"Building TSV from: {src_path}")
    print(f"                 : {tgt_path}")

    kept = 0
    with open(src_path, encoding="utf-8") as src_f, open(tgt_path, encoding="utf-8") as tgt_f:
        with open(output_path, "w", encoding="utf-8") as out_f:
            for i, (src, tgt) in enumerate(zip(src_f, tgt_f)):
                if limit is not None and kept >= limit:
                    break
                src = norm(src)
                tgt = norm(tgt)
                if not src or not tgt:
                    continue
                out_f.write(f"{src}\t{tgt}\n")
                kept += 1

    print(f"Wrote {kept} rows -> {output_path}")
    print(f"Metadata used from -> {metadata_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download and export FLORES-200 Hindi-English TSV.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, choices=["dev", "devtest"], help="FLORES split to export")
    parser.add_argument("--output", default=RAW_OUT, help="Output TSV path")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help="Cache directory for the FLORES tarball and extracted files")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick smoke testing")
    args = parser.parse_args(argv)

    build_hi_en_tsv(
        split=args.split,
        output_path=args.output,
        cache_dir=Path(args.cache_dir),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()