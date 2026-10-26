# Controlled ruler and ink insertion

Completed on the frozen SmolVLM-500M LoRA nevus/melanoma classifier, using its
original test split, prompt, saved 512×512 processor, and 0.5 threshold. No
training or threshold fitting was performed. This is an exploratory follow-up
to the [published-mask removal study](../smolvlm_mask_ablation/README.md).

**Predictions changed substantially even though the annotated lesion and its
surrounding protected box were unchanged. Ink consistently pushed predictions
toward nevus, including for melanomas.** This supports susceptibility to these
synthetic background artifacts. It does not establish how much naturally
occurring artifacts contributed to the original test performance, or whether
LoRA introduced the behavior.

## Diagnostic performance

The common cohort contains **914 images: 830 nevi and 84 melanomas, from 741
lesions**. Each image received three ruler designs and three ink designs, with a
matched control for each, plus an untouched baseline: 11,882 predictions total.
Designs also vary in their deterministic randomized border placement/rotation.

Each table row evaluates all 914 images. Accuracy, sensitivity, and specificity
are percentages; macro-F1 and AUROC are on a 0–1 scale. Sensitivity is melanoma
recall; specificity is nevus recall. These are this subset's results, not a new
full-test score.

| Condition | Accuracy | Macro-F1 | AUROC | Melanoma recall | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original | 79.43 | 0.650748 | 0.874433 | 83.33 | 79.04 |
| Ruler 0 | 77.24 | 0.563935 | 0.715756 | 44.05 | 80.60 |
| Ruler 1 | 78.45 | 0.575850 | 0.699928 | 45.24 | 81.81 |
| Ruler 2 | 77.46 | 0.584795 | 0.750868 | 53.57 | 79.88 |
| Ink 0 | 88.84 | 0.634044 | 0.789508 | 29.76 | 94.82 |
| Ink 1 | 88.51 | 0.621012 | 0.811223 | 27.38 | 94.70 |
| Ink 2 | 88.51 | 0.634187 | 0.786023 | 30.95 | 94.34 |

Ink's higher accuracy is misleading: it removes many nevus false positives
while correctly detected melanomas fall from **70/84 to 23–26/84**. Macro-F1 and
AUROC also fall. Ruler designs flip 23.6–26.9% of predictions; ink designs flip
20.6–21.1%.

## Matched controls and paired effects

Controls permute whole RGBA pixels within each overlay's bounding rectangle.
They preserve its exact native color/opacity histogram and painted pixel count,
at the same location, while destroying its structure. Final composite RGB
histograms need not match, because the underlying skin is nonuniform.

The primary contrast is the **target-minus-control change in the
melanoma-versus-nevus logit margin**, averaging the three designs within each
image before bootstrapping whole lesions. Positive means more melanoma-like;
negative means more nevus-like. Scores are conditional on the A/B answer
choices, not calibrated disease probabilities.

| Artifact | Target − control margin (95% CI) | Conditional melanoma score difference (95% CI), pp |
| --- | ---: | ---: |
| Ruler | +0.841 [0.762, 0.918] | +10.48 [9.37, 11.54] |
| Ink | −1.702 [−1.789, −1.614] | −18.14 [−19.53, −16.70] |

All three designs have the same target-versus-control direction: ruler margin
effects are +0.818 to +0.862; ink effects are −1.797 to −1.536. Each design's
marginal 95% interval excludes zero.

The baseline comparison matters because the controls themselves strongly change
predictions:

| Artifact / diagnosis | Target − original margin (95% CI) | Control − original margin (95% CI) |
| --- | ---: | ---: |
| Ruler / all | +0.149 [0.018, 0.281] | −0.692 [−0.810, −0.573] |
| Ruler / nevus | +0.295 [0.161, 0.429] | −0.557 [−0.677, −0.436] |
| Ruler / melanoma | −1.297 [−1.573, −1.004] | −2.022 [−2.290, −1.743] |
| Ink / all | −1.363 [−1.473, −1.258] | +0.339 [0.267, 0.411] |
| Ink / nevus | −1.289 [−1.400, −1.181] | +0.393 [0.316, 0.469] |
| Ink / melanoma | −2.094 [−2.410, −1.770] | −0.192 [−0.353, −0.025] |

Thus **rulers do not uniformly raise melanoma scores relative to the original**:
they raise them on nevi but lower them on melanomas. Ruler controls reduce
melanoma recall even further, to 23.8–27.4%, and AUROC to 0.737–0.758. A positive
ruler-versus-control contrast alone would obscure this general perturbation
effect. Ink controls retain 71.4–77.4% melanoma recall, substantially above the
27.4–31.0% under structured ink. Ink provides the clearer directional result.

In the annotated training subset, rulers occurred in 71/617 melanomas (11.51%)
and 207/4,665 nevi (4.44%). Ink occurred in only seven nevi and zero annotated
melanomas. These label associations were recorded before scoring. Ink's
direction is compatible with a benign shortcut, but seven noisy labels cannot
establish that these particular training examples caused it. Annotation coverage
also omits 158 training melanomas. The base model without LoRA was not tested.

All confidence intervals use 2,000 whole-lesion bootstrap draws, retaining
correlated views. They are marginal exploratory intervals without multiplicity
adjustment and are conditional on the three fixed designs.

## Frozen ruler-probe manipulation check

The same 16 images selected for visual QA (eight per diagnosis) were specified
before viewing classifier outcomes. The existing frozen ruler probe scored the
original and six ruler/control variants, giving 112 additional predictions. No
probe fitting was performed.

Mean ruler-probe score increased from **0.143 on originals to 0.750 on rulers**,
versus 0.608 on controls. At the saved 0.81 threshold, positive rates were 0%,
58.3%, and 37.5%, respectively. The target-minus-control ruler-probe logit margin
was +1.708 [0.807, 2.733]. This supports a response to the inserted ruler cues,
but the controls activate the readout too. This small check uses an imperfect
probe, not ground truth for synthetic ruler recognition.

## Lesion protection, eligibility, and verification

Lesion masks came from the authors' public
[HAM10000 segmentation archive](https://dataverse.harvard.edu/api/access/datafile/3838943),
curated for [Tschandl et al., Human–computer collaboration for skin cancer
recognition, 2020](https://doi.org/10.1038/s41591-020-0942-0). Its published MD5
`6e8d252e09cfdb0189199f15985a5b84` was verified.

Visual QA found a mask with apparently reversed polarity. No masks were
automatically inverted or repaired. The final protocol conservatively excludes
any foreground at an image corner and protects the **entire foreground bounding
box plus 15 native pixels**, including segmentation holes and gaps. Every target
and control rectangle must avoid this region, and every protected RGB pixel is
asserted unchanged during scoring. Native images are 600×450, then processed
using the saved 512×512 configuration. Protection applies to the annotated index
lesion; segmentation errors and other visible lesions remain limitations.

Exclusions from the original 1,189 test images were determined before inference:

| Reason | Nevus | Melanoma | Total |
| --- | ---: | ---: | ---: |
| Not labelled free of both ink and ruler, including missing labels | 43 | 57 | 100 |
| Earlier published pixel mask confirms pre-existing ink | 6 | 0 | 6 |
| Foreground at image corner: ambiguous polarity or geometry | 55 | 4 | 59 |
| Insufficient clear border for every design | 89 | 21 | 110 |
| Retained | 830 | 84 | 914 |

The corner rule can exclude valid large lesions; the 59 exclusions are **not**
59 confirmed inverted masks. Eligibility retains 81.1% of original test nevi
and 50.6% of melanomas, so generalization to the full cohort is limited.

Plan, script, source-image, source-mask, adapter, and configuration hashes were
recorded. All 5,484 placements and 914 mask hashes were independently checked.
All 11,882 main predictions, diagnostic metrics, primary/class-specific
contrasts, and cluster confidence intervals were independently recomputed after
download. The untouched baseline differs from the historical saved predictions
on one of 914 classifications: mean absolute score difference 0.00193, maximum
0.01558. Paired analyses use originals rescored in this run; numerical differences
across batching/software are a plausible explanation for the small mismatch.

Main scoring and summaries took approximately 13.4 minutes on a Colab T4; the
probe check took 21.5 seconds. The GPU was released after downloading results
and execution provenance.

## Limits and saved files

These are synthetic overlays, not photographs of the same lesion before and
after adding a real artifact. The permutation controls are noise-like and
visually unnatural; resizing can attenuate shuffled pixels differently from
coherent marks. Their strong effects show why control choice matters. Masks
are imperfect, only a sample received visual QA, and the original test set has
already supported exploratory analyses. There is no independent cohort or
clinical review. The evidence establishes susceptibility to the specified
background interventions, not definitive natural-artifact shortcut exploitation.

Useful results in this directory:

- `results.json`, `comparison.csv`, `effects.csv`: full and compact paired metrics.
- `predictions.csv`: all original/target/control classifier outputs and logit margins.
- `cohort.csv`, `placements.csv`, `exclusions.csv`: eligibility, hashes, and exact edits.
- `protocol.json`, `training_artifact_association.csv`: frozen method and label associations.
- `baseline_reproduction.json`: comparison against historical original predictions.
- `ruler_probe_check.csv`, `ruler_probe_check.json`, `ruler_probe_check_plan.json`: supplemental readout check.
- `run_info.json`: command, device, software versions, and elapsed time.

Previews, masks, rejected unscored plans, raw archives, execution notebooks,
logs, and execution scripts stay in ignored `artifacts/`; downloaded lesion
masks stay in ignored `data/`. Reproduce the primary experiment with
`scripts/artifact_insertion.py` using the root README's two-stage commands.
