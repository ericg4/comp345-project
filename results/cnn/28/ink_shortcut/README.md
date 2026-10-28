# Planted ink shortcut: 28px CNN

The 28px CNN (`scripts/cnn.py --size 28`) was retrained with synthetic ink
planted on a fixed share of each class, then tested with the ink kept, removed,
added everywhere, or flipped. Produced by `scripts/ink_shortcut.py` on the
shared split, locally on Apple MPS, in about 9 minutes.

**At 90/10 the CNN takes the shortcut.** On a test set with the same ink bias,
it scores higher than the no-shortcut control (macro-F1 0.814 vs. 0.760). Without
the ink it falls to 0.696, and melanoma recall halves (0.71 → 0.37). At 100/0 the
ink dominates the decision. At 70/30 and 50/50 there is no detectable reliance.

## Setup

- **Ink:** the three ink designs from `scripts/synthetic_artifacts.py`, the same
  graphics as the SmolVLM insertion study. Each image gets one fixed stamp:
  design, rotation (0/90/180/270), and position.
- **Placement:** each stamp is at least 15 native pixels from every lesion-mask
  pixel ([HAM10000 segmentations](https://dataverse.harvard.edu/api/access/datafile/3838943),
  MD5 verified). It also lies inside the central 394×394 region that the CNN's
  center crop keeps. Recomputed after the run: the closest stamp pixel is 15.03 px
  from the lesion, and no stamp falls outside the crop.
- **Difference from the SmolVLM study:** that study protected the lesion's whole
  bounding box plus 15 px. Here that rule left room for ink on only 42% of test
  melanomas, so the lesion's own pixels plus 15 px are protected instead.
- **Exclusions:** images whose mask has foreground at a corner, or with no safe
  position, are excluded everywhere. 6,421 of 7,818 images remain:

| Split | Melanoma | Nevus |
| --- | ---: | ---: |
| Train | 546 / 779 | 3,931 / 4,695 |
| Val | 121 / 166 | 864 / 1,005 |
| Test | 129 / 168 | 830 / 1,005 |

- **Ink assignment:** at level *m*/*n*, exactly *m*% of melanomas and *n*% of nevi
  in train and val carry their stamp. Each image has one fixed random draw, so a
  stronger level inks a superset of the images of a weaker one.
- **Training:** the same training code and settings as `cnn.py --size 28`, with
  three seeds per level. Epoch and threshold are chosen on the equally biased
  val split.
- **Test conditions:**

| Condition | Ink on |
| --- | --- |
| Biased | The level's shares, as in training |
| Clean | No image |
| All | Every image |
| Flipped | The class shares swapped |

- **Baseline:** the existing `data/cnn/28.pt`, trained without ink, scored on the
  clean and all conditions.

## Results

Test means over three seeds (129 melanomas, 830 nevi). Seed-to-seed ranges are
in `summary.csv`.

| Trained at | Macro-F1 biased | clean | all | flipped | Mel. recall biased | clean | AUROC clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (no ink) | n/a | 0.763 | 0.759 | n/a | n/a | 0.659 | 0.885 |
| 50/50 | 0.760 | 0.759 | 0.761 | 0.760 | 0.713 | 0.711 | 0.902 |
| 70/30 | 0.751 | 0.752 | 0.747 | 0.750 | 0.682 | 0.680 | 0.902 |
| 90/10 | **0.814** | **0.696** | 0.711 | 0.613 | 0.680 | **0.367** | 0.900 |
| 100/0 | **0.968** | **0.542** | 0.437 | 0.270 | 0.951 | **0.090** | 0.888 |

Adding ink to every test image shifts the melanoma-minus-nevus logit margin by
the amounts below. Values are per-seed means, with 95% whole-lesion bootstrap
intervals of 2,000 draws.

| Trained at | Melanomas | Nevi |
| --- | --- | --- |
| Baseline | +0.02 [0.01, 0.03] | +0.06 [0.05, 0.07] |
| 50/50 | +0.01 to +0.02 | +0.04 to +0.10 |
| 70/30 | +0.02 to +0.03 | +0.07 to +0.20 |
| 90/10 | +1.28 to +1.50, each CI excludes 0 | +1.62 to +1.76 |
| 100/0 | +6.53 to +7.70 | +7.46 to +8.75 |

## Interpretation

- **The biased score hides the shortcut.** At 90/10 the CNN looks better than the
  no-shortcut control on a test set with the same bias as training, and worse on
  a clean one. That is the gap the shortcut-free test set is meant to expose.
- **There is a threshold between 70/30 and 90/10.** Below it, the faint 28px ink
  (about 8×4 pixels) isn't worth using. Above it, the model relies on it heavily.
- **The lesion features survive. The decision rule is what breaks.** Clean-test
  AUROC stays at 0.90 for 90/10 (0.89 for 100/0), close to the control. The model
  still ranks lesions well, but ink adds a large constant to the melanoma score.
  A threshold tuned on biased data is then badly wrong without ink. A linear probe
  retrained on clean data could likely recover most of the performance, which
  matters when comparing CNN embeddings with PanDerm through probes.
- **Effects are not specific to melanoma.** The margin shift is similar for nevi
  and melanomas.

## Limits

- **One model size.** Only 28px was tested, where the ink is a few pixels across.
  224px may reach the threshold at a weaker correlation.
- **Selected cohort.** Exclusions drop 23% of test melanomas, mostly large lesions
  with no room for ink.
- **Synthetic artifact.** The ink is a synthetic overlay, not real ink.
- **Few seeds.** Three seeds per level; 70/30 seed 2 is the only run below that
  threshold with a visible shift (+0.20 on nevi).

## Files

| File | Contents |
| --- | --- |
| `summary.csv` | One row per model, seed, and test condition |
| `results.json` | Protocol, exclusion counts, per-run test metrics, bootstrap effects, chosen epoch and threshold, training history |
| `predictions.csv.gz` | Per-image probability and logit margin for every run and condition |
| `placements.csv` | Every image's stamp or exclusion reason, plus its draw rank |
| `preview.png` | Two test nevi and two melanomas with ink: native (left) and the 28px crop the CNN sees (right) |
| `run_info.json` | Command, device, versions, and elapsed time |

Model weights are in the ignored `data/cnn/ink_shortcut/`.
