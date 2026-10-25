# Shortcut Learning Project

Do skin lesion classifiers rely on shortcuts? We compare a from-scratch CNN, a
frozen PanDerm encoder with a linear head, and a fine-tuned VLM on HAM10000
nevus vs. melanoma, then probe each model's features for artifacts.

## Files

| File | Purpose |
| --- | --- |
| `splits/ham10000_nv_vs_mel.csv` | Shared 70/15/15 split, grouped by lesion. Every arm uses it. |
| `scripts/common.py` | Shared paths, split loading, and metrics. Every arm imports it. |
| `scripts/make_splits.py` | Creates the split CSV. Already run; re-run only if all arms switch. |
| `scripts/panderm_features.py` | Frozen PanDerm encoder, writes features to `data/features/panderm.npz` |
| `scripts/linear_probe.py` | Trains a linear head on a features file, writes `results/<name>.json` |
| `scripts/download_datasets.py` | Downloads the datasets and PanDerm weights |
| `third_party/PanDerm/` | PanDerm model code (git submodule, pinned commit) |

`data/` (datasets, weights, features) is ignored by git.

## PanDerm linear-probe baseline

```sh
git submodule update --init
uv sync
uv run python scripts/download_datasets.py --ham10000 --panderm
unzip -q data/ham10000/skin-cancer-mnist-ham10000.zip 'HAM10000_images_part_*' \
  HAM10000_metadata.csv -d data/ham10000/skin-cancer-mnist-ham10000
uv run python scripts/panderm_features.py
uv run python scripts/linear_probe.py data/features/panderm.npz
```

Unzip only the upper-case image folders. The Kaggle archive also contains
lower-case duplicates, which collide on case-insensitive filesystems such as macOS.

`panderm_features.py --random-init` writes `panderm_random.npz`: the same
ViT-L/16 without pretrained weights, a control showing how much pretraining adds.

## Skin lesion dataset downloads

Use uv with Python 3.11+ to set up the environment and download any combination
of the datasets. Dependencies are declared in `pyproject.toml` and pinned in
`uv.lock`. `uv sync` creates or updates `.venv`; activation is optional when using
`uv run`.

```sh
uv sync
uv run python scripts/download_datasets.py --ham10000
uv run python scripts/download_datasets.py --scholarsmine --dermartifactdb
uv run python scripts/download_datasets.py --all --extract
uv run python scripts/download_datasets.py --all --dry-run
```

Archives go to `data/<dataset>/` by default. Use `--output-dir /path/to/data`
to change this. `--extract` also unpacks each ZIP into a folder named after the
archive, retaining the ZIP. Extraction overwrites matching files; it does not
remove older extracted files. Allow space for both archives and extracted data.

Existing valid ZIPs are skipped; `--force` downloads fresh copies. Downloads are
streamed into temporary files and moved into place after validation. Interrupted
files restart on the next run; completed archives are retained. The Zenodo archive
is checked against its published MD5. Other ZIPs are checked for readable archive
directories; extraction also checks member CRCs. A failed file does not prevent
the remaining downloads, and any failure produces exit code 1.

| Flag | Source | Contents |
| --- | --- | --- |
| `--ham10000` | [Kaggle: Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | Full Kaggle dataset archive, including images and metadata |
| `--scholarsmine` | [Dermatology Skin Lesion Image Analysis](https://scholarsmine.mst.edu/research_data/15/) | Three supplemental ZIPs: vignetting, hair/ruler, and ink mark masks for ISIC 2019; original ISIC images are separate |
| `--dermartifactdb` | [Zenodo record 18324962](https://zenodo.org/records/18324962) | `DermArtifactDB.zip` (approximately 4.1 GB) |
| `--panderm` | [PanDerm (Google Drive via gdown)](https://github.com/SiyuanYan1/PanDerm) | `panderm_ll_data6_checkpoint-499.pth`, ViT-L/16 weights (approximately 1.4 GB) |

The two public sources require no additional Python packages. HAM10000 uses the
official Kaggle CLI, which requires Python 3.11+ for the current release:

```sh
uv sync
uv run kaggle auth login
uv run python scripts/download_datasets.py --ham10000
```

Alternatively configure `KAGGLE_API_TOKEN` or `~/.kaggle/kaggle.json` as described
in the [official authentication documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#authentication).
Keep credentials outside the repository.

Scholars' Mine returned HTTP 403 to automated requests during source verification.
If this happens, open its source page in a browser and download the three
supplemental files. Save them under `data/scholarsmine/` using the filenames
printed by `--dry-run`; then rerun with `--scholarsmine --extract`.
