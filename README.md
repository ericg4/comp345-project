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
| `scripts/panderm_features.py` | Frozen PanDerm encoder, writes features to `data/features/panderm/pretrained.npz` |
| `scripts/cnn.py` | Trains a small CNN from scratch (`--size` sets the input size), writes `results/cnn/<size>/` and its embeddings to `data/features/cnn/<size>.npz` |
| `scripts/linear_probe.py` | Trains a linear head on a features file, writes `probe.json` to the matching folder (`data/features/a/b.npz` → `results/a/b/`) |
| `scripts/train_smolvlm.py` | Trains language LoRA adapters with the original vocabulary head on the shared split |
| `scripts/predict_smolvlm.py` | Predicts images using saved LoRA adapters, including the preserved Colab run |
| `results/` | Results to commit: metrics, predictions, configuration, and split records |
| `artifacts/` | Gitignored model weights, tokenizers, archives, and execution records |
| `scripts/download_datasets.py` | Downloads the datasets and PanDerm weights |
| `third_party/PanDerm/` | PanDerm model code (git submodule, pinned commit) |

`data/` (datasets, PanDerm weights, features) and `artifacts/` (generated model
files and other run side products) are ignored by git. `results/` stays trackable.

## PanDerm linear-probe baseline

```sh
git submodule update --init
uv sync
uv run python scripts/download_datasets.py --ham10000 --panderm
unzip -q data/ham10000/skin-cancer-mnist-ham10000.zip 'HAM10000_images_part_*' \
  HAM10000_metadata.csv -d data/ham10000/skin-cancer-mnist-ham10000
uv run python scripts/panderm_features.py
uv run python scripts/linear_probe.py data/features/panderm/pretrained.npz
```

Unzip only the upper-case image folders. The Kaggle archive also contains
lower-case duplicates, which collide on case-insensitive filesystems such as macOS.

`panderm_features.py --random-init` writes `panderm/random_init.npz`: the same
ViT-L/16 without pretrained weights, a control showing how much pretraining adds.

## SmolVLM LoRA

Use a CUDA GPU (including Colab via its CLI). After downloading and extracting
HAM10000 as above:

```sh
uv sync --extra vlm
uv run python scripts/train_smolvlm.py --epochs 5
```

The script uses `common.load_split()` and `common.metrics()`, with the committed
shared split. It keeps SmolVLM-500M's pretrained vocabulary head, trains only
language attention LoRA adapters, and predicts `A` (nevus) or `B` (melanoma).
Images are resized to 512×512 without augmentation. Defaults are rank 8,
microbatch 8, accumulation 2, and learning rate 1e-4. FP16 is used on a T4;
newer compatible GPUs use BF16.

Useful results go to `results/smolvlm_lora/`: the exact split, configuration,
epoch history, test metrics, and per-image predictions. Model weights, tokenizer,
and a copy of the inference configuration go to the gitignored
`artifacts/smolvlm_lora/`. Existing folders are rejected to protect previous runs.
Use `--output-dir results/my_run` for another run; its model files default to
`artifacts/my_run/`. `--artifacts-dir` can override that location.
`--data-dir` and `--split-csv` accept alternate dataset and split paths. Epoch
selection uses validation average precision, with a fixed decision threshold of
0.5. PanDerm's linear probe instead selects by validation macro-F1 and tunes its
threshold; keep this difference in mind when comparing the arms.

The earlier Colab run is saved in
[`results/ham10000_lora/`](results/ham10000_lora/README.md). **It used a different
split**, so its reported scores are historical results, not a comparison on the
current shared split. Its weights and results have not been overwritten. The
notebook has been replaced by the training script.

To predict with the saved original run (downloads the frozen base model):

```sh
uv run python scripts/predict_smolvlm.py artifacts/ham10000_lora path/to/image.jpg
```

The reported melanoma score is conditional on the two answer tokens, not a
calibrated disease probability. Inference supports CUDA, Apple MPS, or CPU.

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
