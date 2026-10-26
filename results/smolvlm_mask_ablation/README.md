# Published-mask intervention results

Completed on the frozen, previously trained SmolVLM-500M LoRA lesion classifier.
The prompt, saved 512×512 processor, original test split, and 0.5 decision
threshold were held fixed. No model training or threshold tuning was performed.

Three published mask types from [Wegley et al., Dermatology Skin Lesion Image
Analysis](https://scholarsmine.mst.edu/research_data/15/) were tested:
**hair/ruler together**, ink, and vignetting. The first mask marks the union of
hair and ruler; it cannot isolate their effects. Selection used the pixel masks
directly, not DermArtifactDB's image-level presence labels.

## Paired accuracy results

Each row uses its own covered test subset. Values are accuracy percentages;
positive drops mean worse accuracy after editing. Do not compare raw accuracy
across these differently balanced subsets as a ranking of artifact importance.

| Mask type | Images (lesions) | Nevus / melanoma | Original | Occluded | Inpainted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hair/ruler union | 1,170 (896) | 1,006 / 164 | 78.80% | 72.39% | 72.91% |
| Ink | 6 (6) | 6 / 0 | 33.33% | 33.33% | 50.00% |
| Vignetting | 102 (98) | 68 / 34 | 59.80% | 65.69% | 46.08% |

| Mask type | Occlusion drop, pp (95% CI) | Inpainting drop, pp (95% CI) |
| --- | ---: | ---: |
| Hair/ruler union | 6.41 [3.93, 8.91] | 5.90 [3.79, 8.08] |
| Ink | 0.00 [-50.00, 50.00] | -16.67 [-50.00, 0.00] |
| Vignetting | -5.88 [-15.85, 4.00] | 13.73 [5.94, 21.78] |

Intervals are percentile bootstrap intervals from 2,000 resamples of whole
lesions, keeping repeated views together. They are exploratory per-comparison
intervals without multiplicity adjustment. Ink has only six nevi: its accuracy
is measurable, but macro-F1, melanoma F1, balanced accuracy, and AUROC are not
reported for this single-class subset. These six cases do not support a broad
conclusion about ink or melanoma detection. Unchanged ink occlusion accuracy
also hides two prediction flips: one improvement and one regression.

## Same-area controls

Each control is the same mask circularly translated to minimize overlap with
the target foreground among 64 seeded candidates. It has exactly the same pixel
count and uses the same fill or inpainting method. Mean target-mask area is
6.40% for hair/ruler, 3.95% for ink, and 3.61% for vignetting. Mean remaining
foreground overlap is 3.63%, 0.009%, and 0%, respectively.

| Mask type | Control-occluded accuracy | Control-inpainted accuracy |
| --- | ---: | ---: |
| Hair/ruler union | 70.94% | 76.92% |
| Ink | 16.67% | 16.67% |
| Vignetting | 57.84% | 60.78% |

Additional drop beyond control is **control accuracy minus target-edited
accuracy**. Its intervals use the directly paired target/control outcomes,
not subtraction of separately estimated confidence limits.

| Mask type | Additional occlusion drop, pp (95% CI) | Additional inpainting drop, pp (95% CI) |
| --- | ---: | ---: |
| Hair/ruler union | -1.45 [-3.97, 1.08] | 4.02 [2.16, 5.98] |
| Ink | -16.67 [-50.00, 0.00] | -33.33 [-66.67, 0.00] |
| Vignetting | -7.84 [-19.61, 3.96] | 14.71 [6.00, 23.30] |

## Macro-F1 and class separation

| Mask type | Original macro-F1 | Occluded macro-F1 | Inpainted macro-F1 |
| --- | ---: | ---: | ---: |
| Hair/ruler union | 0.694245 | 0.638412 | 0.650392 |
| Ink | Not defined | Not defined | Not defined |
| Vignetting | 0.598001 | 0.649278 | 0.454439 |

Hair/ruler inpainting increased mean melanoma score by 0.0668. Nevus false
positives increased from 221 to 302, while correctly detected melanomas also
increased from 137 to 149. AUROC **improved from 0.875852 to 0.891005**, and
balanced accuracy changed little (0.807842 to 0.804169). Its ordinary accuracy
loss is therefore largely a shift toward melanoma at the fixed threshold on a
nevus-heavy subset, rather than evidence of worse class ranking.

Vignetting inpainting reduced balanced accuracy from 0.676471 to 0.558824 and
AUROC from 0.825908 to 0.784170. Literal occlusion instead improved ordinary
accuracy, illustrating the dependence on how the region is replaced.

## Interpretation

Predictions are sensitive to the edited regions, but **these results do not
establish that the model is "cheating."** For hair/ruler, plain occlusion hurt
less than its matched control, while target inpainting hurt more than control
inpainting. Vignetting also depended strongly on the replacement method.

Inpainting reconstructs unknown content and can change texture; published
semi-automatic masks can cover legitimate lesion information. Translated masks
change spatial context: for example, a vignette-corner control can land inside
the lesion. They help assess general editing damage but are not a perfect
artifact-only counterfactual. Hair/ruler remains a combined effect, and the ink
subset is too small for a reliable general claim. The evidence supports
artifact-region/operating-point sensitivity and a more controlled follow-up,
rather than a definitive shortcut-use conclusion.

## Protocol, coverage, and saved files

The source is the original model's 1,189 held-out images. Native image/mask
dimensions matched exactly at 600×450. Grayscale boundaries were thresholded
at 128; binary masks retained their meaning. Opaque, nearly grayscale RGB/RGBA
masks were converted to luminance. One native-image pixel of dilation was
applied. Occlusion used median RGB outside the target mask, identical for its
control. Inpainting used OpenCV Telea with radius 3. The saved processor then
performed the same resizing/normalization as normal inference.

Coverage before scoring:

- Hair/ruler: all 1,189 masks matched; 19 empty masks were excluded.
- Ink: 6 matched masks were nonempty; 1,183 images lacked a matching mask.
- Vignetting: 104 matched masks; 2 empty and 1,085 missing masks were excluded.

Missing masks were never treated as proof of artifact absence. Selection and
editing settings were fixed before scoring. All 6,390 predictions, paired
contrasts, confidence intervals, model/split hashes, and 1,278 source-mask hashes
were verified after download. Original classifier and probe results were not
overwritten. The Colab T4 was released after the approximately 9.7-minute run.

Useful committed files:

- `results.json`: complete method metadata and paired metrics/contrasts.
- `comparison.csv`: compact metrics with percentage-point accuracy changes.
- `predictions.csv`: original and four edited-condition predictions per case.
- `cohort.csv`: cases, labels, source-mask hashes, areas, and control overlaps.
- `excluded_masks.csv`: per-mask missing/empty exclusion audit.
- `run_info.json`: exact command, runtime, device, and software versions.

Ignored side products are under `artifacts/smolvlm_mask_ablation/`: preview
grids, target masks, and control masks. Preview columns are original, occluded,
control-occluded, inpainted, control-inpainted. The job bundle, output archive,
execution notebook, and log remain under ignored `artifacts/`.

Reproduce using the root README's published-mask commands. Existing result
folders are protected; use a new `--output-dir` for reruns.
