import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import ssl
import tempfile
import urllib.request
import zipfile

try:
    import certifi
except ImportError:
    certifi = None

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASETS = {
    "ml-100k": ("https://files.grouplens.org/datasets/movielens/ml-100k.zip", 100 * 1024 * 1024),
    "ml-1m": ("https://files.grouplens.org/datasets/movielens/ml-1m.zip", 200 * 1024 * 1024),
}


def _safe_members(archive, dataset, max_size):
    members, total_size = [], 0
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        mode = info.external_attr >> 16
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != dataset:
            raise ValueError(f"unsafe archive member: {name!r}")
        if len(path.parts) > 2 or (len(path.parts) == 2 and path.parts[1] in ("", ".")):
            raise ValueError(f"unexpected archive layout: {name!r}")
        if mode and (mode & 0o170000) not in (0, 0o100000, 0o040000):
            raise ValueError(f"unsupported special archive member: {name!r}")
        total_size += info.file_size
        if total_size > max_size:
            raise ValueError("archive exceeds the uncompressed size limit")
        members.append(info)
    return members


def download(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}; choose one of {sorted(DATASETS)}")
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / dataset
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {target}")

    url, max_size = DATASETS[dataset]
    archive_path = None
    stage_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f"{dataset}-", suffix=".zip", dir=RAW, delete=False) as archive_file:
            archive_path = Path(archive_file.name)
            request = urllib.request.Request(url, headers={"User-Agent": "personality-movie-recommender/0.1"})
            context = ssl.create_default_context(cafile=certifi.where()) if certifi else None
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                shutil.copyfileobj(response, archive_file)
        stage_path = Path(tempfile.mkdtemp(prefix=f".{dataset}-extract-", dir=RAW))
        with zipfile.ZipFile(archive_path) as archive:
            members = _safe_members(archive, dataset, max_size)
            archive.extractall(stage_path, members=members)
        extracted = stage_path / dataset
        if not extracted.is_dir():
            raise ValueError("archive did not contain expected dataset directory")
        if target.exists():
            raise FileExistsError(f"refusing to overwrite path created during download: {target}")
        os.replace(extracted, target)
        return target
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        if stage_path is not None:
            shutil.rmtree(stage_path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download an official MovieLens 100K or 1M dataset.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    args = parser.parse_args()
    print(f"Downloaded to {download(args.dataset)}")
