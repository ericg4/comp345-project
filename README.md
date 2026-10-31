# Shortcut learning in skin lesion classifiers

Do melanoma classifiers look at the lesion, or at artifacts around it such as ink
marks and rulers? We train three kinds of model on HAM10000 nevus vs. melanoma
(a small CNN from scratch, a linear probe on the frozen PanDerm foundation model,
and SmolVLM-500M fine-tuned with LoRA), then add, remove, and plant artifacts to
see how their predictions change.

**Start with [`00_demo.ipynb`](00_demo.ipynb).** It runs on CPU in seconds using
only the files in `demo/`.

## Running the demo

The demo needs [uv](https://docs.astral.sh/uv/) and nothing else: no dataset
download, no PanDerm weights, no GPU, and no `git submodule` step. From the repo root:

```sh
uv sync --extra demo
uv run jupyter notebook 00_demo.ipynb    # or: uv run jupyter lab
```

Then run all cells (Kernel → Restart & Run All). The notebook loads files by
relative path, so start Jupyter from the repo root. In VS Code, pick the
`.venv` interpreter created by `uv sync` as the kernel. To run it without a browser:

```sh
uv run --extra demo jupyter nbconvert --to notebook --execute 00_demo.ipynb --output 00_demo.ipynb
```

What it does:

1. **Live:** the 28px CNN classifies 6 test images (`demo/images/`, `demo/samples.csv`).
2. **Live:** synthetic ink is stamped beside each lesion, and the images are classified
   again by the normal CNN and by a CNN trained with a 90/10 ink bias
   (`demo/weights/`).
3. **Cached:** PanDerm and SmolVLM results are read from `results/`; those models
   are not run.

## Results so far

| Model (clean test split) | Macro-F1 | AUROC |
| --- | ---: | ---: |
| CNN, 28px | 0.764 | 0.888 |
| CNN, 224px | 0.764 | 0.913 |
| PanDerm linear probe | **0.855** | **0.953** |
| PanDerm architecture, random weights, linear probe | 0.747 | 0.890 |
| SmolVLM + LoRA (different split*) | 0.691 | 0.874 |

| Experiment | Finding |
| --- | --- |
| [Planted ink shortcut, CNN](results/cnn/28/ink_shortcut/) | When 90% of training melanomas (and 10% of nevi) carry ink, the CNN uses it: without ink, melanoma recall falls from 0.66 to 0.37. |
| [Planted ink shortcut, PanDerm probe](results/panderm/pretrained/ink_shortcut/) | Same setup: the frozen-PanDerm probe takes the shortcut already at 70/30, and at 90/10 its clean melanoma recall falls to 0.26. |
| [Ink/ruler insertion, SmolVLM](results/smolvlm_artifact_insertion/) | Ink placed beside the lesion drops melanoma recall from 0.83 to 0.29. |
| [Artifact probes, SmolVLM](results/smolvlm_artifact_probes/) | Hair (AUROC 0.87) and rulers (0.93) are linearly decodable from its decision feature. |
| [Artifact removal, SmolVLM](results/smolvlm_mask_ablation/) | Inpainting hair/ruler regions costs 4.0 pp accuracy beyond a same-area control. |

\*SmolVLM was trained on an earlier lesion-grouped split. All other models share
`splits/ham10000_nv_vs_mel.csv` (70/15/15, grouped by lesion). The scripts for the
three SmolVLM artifact studies are not in this repo; only their results are.

## Layout

| Path | Contents |
| --- | --- |
| `00_demo.ipynb` | Demo: live CNN predictions with and without ink, plus cached results |
| `demo/` | Demo inputs: 6 test images, ink positions, two 28px CNN checkpoints |
| `src/common.py` | Paths, split loading, metrics, shared helpers |
| `src/download.py` | Downloads datasets and PanDerm weights into `data/` |
| `src/make_split.py` | Creates the shared split (already committed) |
| `src/cnn.py` | Trains the CNN; saves results and embeddings |
| `src/panderm.py` | Extracts frozen PanDerm features |
| `src/linear_probe.py` | Trains a linear head on saved features |
| `src/smolvlm.py` | SmolVLM LoRA training and prediction |
| `src/artifacts.py` | Synthetic ink and ruler overlays |
| `src/ink_shortcut.py` | Planted ink shortcut experiment (CNN) |
| `src/panderm_ink_shortcut.py` | The same experiment for the PanDerm linear probe |
| `results/` | Metrics and test predictions for every experiment |
| `third_party/PanDerm/` | PanDerm model code (git submodule) |

`data/` (datasets, weights, features) is gitignored.

## Reproducing

```sh
git submodule update --init
uv sync
uv run python src/download.py --ham10000 --panderm --ham10000-segmentations --extract
```

HAM10000 needs Kaggle API credentials. If the Kaggle archive's lower-case duplicate
folders cause trouble on macOS, unzip only `HAM10000_images_part_*` and
`HAM10000_metadata.csv`.

```sh
uv run python src/cnn.py --size 28                      # also --size 224
uv run python src/linear_probe.py data/features/cnn/28.npz
uv run python src/panderm.py                            # --random-init for the control
uv run python src/linear_probe.py data/features/panderm/pretrained.npz
uv run python src/ink_shortcut.py                       # about 10 min on Apple MPS
uv run python src/panderm_ink_shortcut.py               # about 30 min on Apple MPS
uv sync --extra vlm && uv run python src/smolvlm.py train   # CUDA GPU
```
