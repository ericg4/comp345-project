# Inserting synthetic rulers and ink (SmolVLM)

The frozen SmolVLM classifier scored 914 test images (830 nevi, 84 melanomas)
with each of the three ruler and ink designs from `src/artifacts.py` pasted
outside the lesion's bounding box + 15 px. Each insertion has a matched control:
the same pixels shuffled (same color and area, no shape).

| Condition | Accuracy | Macro-F1 | AUROC | Melanoma recall | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 79.4 | 0.651 | 0.874 | 83.3 | 79.0 |
| Ruler (3 designs) | 77.2–78.5 | 0.564–0.585 | 0.700–0.751 | 44.0–53.6 | 79.9–81.8 |
| Ink (3 designs) | 88.5–88.8 | 0.621–0.634 | 0.786–0.811 | 27.4–31.0 | 94.3–94.8 |

**Ink pushes predictions toward nevus:** melanoma recall falls from 70/84 to
23–26/84, even though the lesion pixels are untouched. Accuracy rises only
because the test set is mostly nevi. Relative to the shuffled control, the
melanoma-vs-nevus logit margin moves −1.70 [−1.79, −1.61] for ink and
+0.84 [0.76, 0.92] for rulers (`effects.csv`). Ruler *controls* also hurt
melanoma recall, so the ruler effect is partly generic image damage; ink is the
clear result. In training, ink appears on only 7 nevi and 0 melanomas.

The script for this study is not in the repo.
