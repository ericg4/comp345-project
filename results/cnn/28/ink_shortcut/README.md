# Planted ink shortcut (28px CNN)

`src/ink_shortcut.py` retrains the 28px CNN with synthetic ink on a fixed share of
each class in train and val (e.g. 90/10 = 90% of melanomas, 10% of nevi), with 3
seeds per level. Ink is stamped at least 15 px from every lesion-mask pixel and
inside the center crop. Images with no safe position are excluded everywhere,
leaving 6,421 of 7,818 (test: 129 melanomas, 830 nevi). Test conditions:
**biased** (same shares as training), **clean** (no ink), **all** (ink on every
image), **flipped** (shares swapped).

Test means over 3 seeds (`summary.csv` has every run):

| Trained at | Macro-F1 biased | clean | all | flipped | Mel. recall biased | clean | AUROC clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (no ink) | n/a | 0.763 | 0.759 | n/a | n/a | 0.659 | 0.885 |
| 50/50 | 0.760 | 0.759 | 0.761 | 0.760 | 0.713 | 0.711 | 0.902 |
| 70/30 | 0.751 | 0.752 | 0.747 | 0.750 | 0.682 | 0.680 | 0.902 |
| 90/10 | **0.814** | **0.696** | 0.711 | 0.613 | 0.680 | **0.367** | 0.900 |
| 100/0 | **0.968** | **0.542** | 0.437 | 0.270 | 0.951 | **0.090** | 0.888 |

- **The CNN takes the shortcut at 90/10.** On a test set biased like training it
  beats the ink-free baseline; on a clean one it falls below it and melanoma
  recall halves. At 70/30 and below there is no detectable reliance.
- **Ranking survives, the threshold breaks.** Clean AUROC stays about 0.90: ink
  adds a roughly constant amount to the melanoma score (+1.3 to +1.8 logits at
  90/10, for both classes), so a threshold tuned on biased data fails on clean data.

Limits: one input size, a synthetic artifact, 3 seeds, and exclusions that drop
23% of test melanomas (mostly large lesions with no room for ink).
