# Planted ink shortcut (PanDerm linear probe)

`src/panderm_ink_shortcut.py` repeats the [CNN ink shortcut](../../../cnn/28/ink_shortcut/)
with a linear probe on frozen PanDerm features. It uses the same 6,421 images, ink
stamps, levels, and test conditions (test: 129 melanomas, 830 nevi). The encoder
never trains, so the probe can only use ink if ink is linearly readable in
PanDerm's features. The baseline probe is trained without ink. One seed (0).

| Trained at | Macro-F1 biased | clean | all | flipped | Mel. recall biased | clean | AUROC clean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline (no ink) | n/a | 0.810 | 0.713 | n/a | n/a | 0.651 | 0.941 |
| 50/50 | 0.817 | 0.812 | 0.816 | 0.817 | 0.721 | 0.752 | 0.949 |
| 70/30 | 0.838 | **0.756** | 0.817 | 0.763 | 0.667 | **0.426** | 0.943 |
| 90/10 | **0.896** | **0.677** | 0.727 | 0.561 | 0.806 | **0.264** | 0.921 |
| 100/0 | **0.989** | **0.478** | 0.136 | 0.018 | 0.984 | **0.016** | 0.703 |

Change in the melanoma-minus-nevus logit margin when ink is added to every test
image (95% whole-lesion bootstrap):

| Trained at | Melanomas | Nevi |
| --- | --- | --- |
| Baseline | −1.54 [−1.66, −1.42] | −1.52 [−1.57, −1.48] |
| 50/50 | −0.30 [−0.36, −0.24] | −0.21 [−0.23, −0.19] |
| 70/30 | +1.28 [1.17, 1.41] | +1.15 [1.10, 1.19] |
| 90/10 | +4.06 [3.82, 4.30] | +3.40 [3.30, 3.50] |
| 100/0 | +13.20 [12.32, 14.05] | +15.19 [14.88, 15.49] |

- **PanDerm takes the shortcut sooner and harder than the CNN.** Reliance starts at
  70/30, where the 28px CNN showed none, and at 90/10 the margin shift is 2–3×
  the CNN's. At 224px the ink is about 64×32 pixels (vs. about 8×4 for the
  CNN), and the frozen features encode it linearly.
- **Ink also moves the no-ink baseline**, toward nevus: adding ink halves its
  melanoma recall (0.651 → 0.333), while AUROC barely moves (0.941 → 0.935). The
  SmolVLM insertion study showed the same direction.
- **Up to 90/10, ranking survives and the threshold breaks**, as with the CNN
  (clean AUROC ≥ 0.92). At 100/0 the ranking also degrades (0.703).

Limits: one seed, a synthetic artifact, and a cohort that excludes 23% of test
melanomas. The clean baseline (0.810) is therefore below the full-split probe (0.855).
