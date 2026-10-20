#!/usr/bin/env python3
"""Download selected skin lesion datasets (Python 3.9+, standard library)."""

import argparse
import hashlib
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
SCHOLARSMINE_BASE = (
    "https://scholarsmine.mst.edu/cgi/viewcontent.cgi?"
    "article=1015&context=research_data&type=additional&filename="
)
# Names follow the supplemental-file order on research_data/15/.
FILES = {
    "scholarsmine": [
        ("ISIC_2019_Vignetting_Masks.zip", SCHOLARSMINE_BASE + "0", None),
        ("ISIC_2019_Hair_Ruler_Masks.zip", SCHOLARSMINE_BASE + "1", None),
        ("ISIC_2019_Inkmark_Masks.zip", SCHOLARSMINE_BASE + "2", None),
    ],
    "dermartifactdb": [
        (
            "DermArtifactDB.zip",
            "https://zenodo.org/records/18324962/files/DermArtifactDB.zip?download=1",
            "79c845cbe6bfb08be9e5e2455fd9c909",
        ),
    ],
}
CHUNK_SIZE = 1024 * 1024


def validate_archive(path, checksum=None):
    if checksum:
        digest = hashlib.md5()  # Publisher-provided integrity checksum.
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                digest.update(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Checksum mismatch: {path}")
    # Reading the directory catches HTML error pages and truncated archives
    # without decompressing several GB just to check a completed download.
    with zipfile.ZipFile(path) as archive:
        if not archive.infolist():
            raise ValueError(f"Empty archive: {path}")


def download_file(url, destination, checksum=None, force=False):
    if destination.exists() and not force:
        validate_archive(destination, checksum)
        print(f"Already downloaded: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so the final rename is atomic.
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as tmp:
        partial = Path(tmp) / destination.name
        request = Request(url, headers={"User-Agent": "skin-lesion-dataset-downloader/1.0"})
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            if "text/html" in response.headers.get("Content-Type", "").lower():
                raise ValueError(f"Server returned an HTML page instead of a ZIP: {url}")
            expected = response.headers.get("Content-Length")
            total = int(expected) if expected else None
            received = 0
            last_update = time.monotonic()
            for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                output.write(chunk)
                received += len(chunk)
                if time.monotonic() - last_update >= 5:
                    progress = f"{received / 1024**2:.1f} MiB"
                    if total:
                        progress += f" / {total / 1024**2:.1f} MiB"
                    print(f"  {destination.name}: {progress}", flush=True)
                    last_update = time.monotonic()
            if total is not None and received != total:
                raise ValueError(f"Incomplete download: expected {total} bytes, got {received}")
        validate_archive(partial, checksum)
        partial.replace(destination)
    print(f"Saved: {destination}", flush=True)


def download_ham10000(destination, force=False):
    if destination.exists() and not force:
        validate_archive(destination)
        print(f"Already downloaded: {destination}", flush=True)
        return
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError(
            "HAM10000 requires the Kaggle CLI. Install with 'uv sync', "
            "then configure credentials (see README.md)."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kaggle-", dir=destination.parent) as tmp:
        subprocess.run(
            [kaggle, "datasets", "download", "-d", KAGGLE_DATASET, "-p", tmp],
            check=True,
        )
        archive = Path(tmp) / destination.name
        validate_archive(archive)
        archive.replace(destination)
    print(f"Saved: {destination}", flush=True)


def extract_archive(path):
    destination = path.parent / path.stem
    root = destination.resolve()
    with zipfile.ZipFile(path) as archive:
        # Validate every member before writing anything. Also reject paths
        # traversing existing symlinks in the extraction directory.
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive contains a symlink: {member.filename}")
        archive.extractall(destination)
    print(f"Extracted: {destination}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ham10000", action="store_true", help="Kaggle HAM10000 images and metadata")
    parser.add_argument("--scholarsmine", action="store_true", help="All three ISIC 2019 artifact mask ZIPs")
    parser.add_argument("--dermartifactdb", action="store_true", help="Zenodo DermArtifactDB (record 18324962)")
    parser.add_argument("--all", action="store_true", help="Download all three datasets")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data",
                        help="Destination directory (default: project data/)")
    parser.add_argument("--extract", action="store_true", help="Extract ZIPs into folders named after each archive; retain ZIPs")
    parser.add_argument("--force", action="store_true", help="Download again, replacing existing ZIPs after validation")
    parser.add_argument("--dry-run", action="store_true", help="Show selection and destinations without downloading")
    args = parser.parse_args(argv)
    selected = [name for name in ("ham10000", "scholarsmine", "dermartifactdb")
                if args.all or getattr(args, name)]
    if not selected:
        parser.error("select --ham10000, --scholarsmine, --dermartifactdb, or --all")
    failures = 0
    for name in selected:
        files = ([("skin-cancer-mnist-ham10000.zip", f"kaggle:{KAGGLE_DATASET}", None)]
                 if name == "ham10000" else FILES[name])
        for filename, url, checksum in files:
            destination = args.output_dir.expanduser() / name / filename
            print(f"{name}: {url}\n  -> {destination}", flush=True)
            if args.dry_run:
                continue
            try:
                if name == "ham10000":
                    download_ham10000(destination, args.force)
                else:
                    download_file(url, destination, checksum, args.force)
                if args.extract:
                    extract_archive(destination)
            except (OSError, URLError, ValueError, RuntimeError,
                    subprocess.CalledProcessError, zipfile.BadZipFile) as error:
                failures += 1
                print(f"ERROR [{name}/{filename}]: {error}", file=sys.stderr)
                if name == "scholarsmine":
                    print("If access is denied, download the supplemental ZIPs from "
                          "https://scholarsmine.mst.edu/research_data/15/ "
                          "and save with the destination names above.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run again to retry unfinished files.", file=sys.stderr)
        sys.exit(130)
