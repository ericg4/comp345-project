# Removing real artifacts from images (SmolVLM)

The frozen SmolVLM classifier scored test images with published artifact masks
([Wegley et al.](https://scholarsmine.mst.edu/research_data/15/)) either occluded
(median fill) or inpainted (OpenCV Telea). A same-area control moves the mask off
the artifact to separate "artifact removed" from "image edited". Intervals are
95% whole-lesion bootstrap (2,000 draws).

| Mask | Images | Original acc. | Inpainted acc. | Inpainting drop beyond control, pp |
| --- | ---: | ---: | ---: | ---: |
| Hair + ruler | 1,170 | 78.8% | 72.9% | +4.0 [2.2, 6.0] |
| Vignetting | 102 | 59.8% | 46.1% | +14.7 [6.0, 23.3] |
| Ink | 6 | 33.3% | 50.0% | too few images |

Predictions are sensitive to the artifact regions, but the result depends on how
the region is filled: occluding hair/ruler hurt *less* than its control (−1.5 pp
[−4.0, 1.1]). The hair/ruler mask is one combined mask, so their effects can't be
separated. Hair/ruler inpainting mostly shifted scores toward melanoma (AUROC
actually rose, 0.876 → 0.891). Evidence of sensitivity, not proof of a shortcut.

The script for this study is not in the repo.
