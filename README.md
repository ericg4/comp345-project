# Shortcut Learning Project

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
