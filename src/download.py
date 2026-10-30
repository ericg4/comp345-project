"""Download the datasets and PanDerm weights into data/.

    uv run python src/download.py --ham10000 --panderm
    uv run python src/download.py --all --extract
"""

import argparse
import hashlib
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from common import DATA_DIR

SCHOLARSMINE = "https://scholarsmine.mst.edu/cgi/viewcontent.cgi?article=1015&context=research_data&type=additional&filename="
# name: [(filename, url, md5 or None)]
FILES = {
    "ham10000": [("skin-cancer-mnist-ham10000.zip", "kaggle:kmader/skin-cancer-mnist-ham10000", None)],
    "panderm": [("panderm_ll_data6_checkpoint-499.pth", "gdrive:1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE", None)],
    "ham10000_segmentations": [("HAM10000_segmentations_lesion_tschandl.zip",
                                "https://dataverse.harvard.edu/api/access/datafile/3838943",
                                "6e8d252e09cfdb0189199f15985a5b84")],
    "dermartifactdb": [("DermArtifactDB.zip", "https://zenodo.org/records/18324962/files/DermArtifactDB.zip?download=1",
                        "79c845cbe6bfb08be9e5e2455fd9c909")],
    "scholarsmine": [("ISIC_2019_Vignetting_Masks.zip", SCHOLARSMINE + "0", None),
                     ("ISIC_2019_Hair_Ruler_Masks.zip", SCHOLARSMINE + "1", None),
                     ("ISIC_2019_Inkmark_Masks.zip", SCHOLARSMINE + "2", None)],
}


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, destination):
    if url.startswith("kaggle:"):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["kaggle", "datasets", "download", "-d", url.removeprefix("kaggle:"), "-p", tmp], check=True)
            next(Path(tmp).glob("*.zip")).replace(destination)
    elif url.startswith("gdrive:"):
        import gdown
        gdown.download(id=url.removeprefix("gdrive:"), output=str(destination), quiet=False)
    else:
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response, \
                destination.open("wb") as f:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                f.write(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in FILES:
        parser.add_argument(f"--{name.replace('_', '-')}", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--extract", action="store_true", help="Unzip each archive into a folder of the same name")
    args = parser.parse_args()
    selected = [name for name in FILES if args.all or getattr(args, name)]
    if not selected:
        parser.error("choose at least one dataset, or --all")

    for name in selected:
        for filename, url, checksum in FILES[name]:
            destination = DATA_DIR / name / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                print(f"Already downloaded: {destination}")
            else:
                print(f"Downloading {destination}")
                download(url, destination)
            if checksum and md5(destination) != checksum:
                raise ValueError(f"Checksum mismatch: {destination}; delete it and retry")
            if args.extract and destination.suffix == ".zip":
                with zipfile.ZipFile(destination) as archive:
                    archive.extractall(destination.with_suffix(""))
                print(f"Extracted {destination.with_suffix('')}")


if __name__ == "__main__":
    main()
