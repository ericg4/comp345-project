# Artifact probes on the trained SmolVLM

Three binary readout tasks were requested: hair/no hair, ink/no ink, and
ruler/no ruler. Hair and ruler were fitted and evaluated. Ink was skipped because
the unchanged test split contains no ink-positive annotations.

## Model and method

The original SmolVLM-500M LoRA checkpoint from `results/ham10000_lora/` was
loaded and fully frozen. Each image used the saved 512×512 processor and the
original lesion-classification prompt. Features are the 960-dimensional final
prompt token immediately before the original vocabulary head. No artifact
questions, labels, heatmaps, or diagnosis labels were added to the model input.

Each task uses an independent class-weighted L2 logistic regression. Scaling
uses training data only. C was selected from `[0.0001, 0.001, 0.01, 0.1, 1, 10]`;
thresholds from 0.05 through 0.95 were selected by validation macro-F1. The test
set was evaluated only after those choices. Both selected C=0.01; hair selected
threshold 0.49 and ruler selected 0.81.

## Data and coverage

Annotations: Gómez-Martínez, Chushig-Muzo, and Soguero-Ruiz,
[DermArtifactDB, Zenodo record 18324962](https://zenodo.org/records/18324962),
ISIC-2019 CSV. The archive's README licenses the annotations under CC BY 4.0.
Labels combine automated annotation and expert review and are not perfect
ground truth. Original image licenses remain applicable.

The original model's image/lesion split was retained. 7,597 of its 7,818 images
have annotations: 5,282 train, 1,163 validation, and 1,152 test. Missing labels
were excluded, never treated as negative. **All 221 excluded images are
melanomas:** 158 train, 26 validation, and 37 test. The annotated test subset has
1,023 nevi and 129 melanomas. Results apply to this subset.

| Artifact | Train positives | Validation positives | Test positives | Test negatives |
| --- | ---: | ---: | ---: | ---: |
| Hair | 3,269 | 715 | 707 | 445 |
| Ink | 7 | 2 | 0 | 1,152 |
| Ruler | 278 | 68 | 63 | 1,089 |

## Held-out results

| Task | Macro-F1 | Artifact-present F1 | AUROC | Average precision | Majority baseline macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hair | 0.789181 | 0.821938 | 0.871564 | 0.917763 | 0.380312 |
| Ruler | 0.710698 | 0.454545 | 0.926290 | 0.518685 | 0.485944 |
| Ink | Not evaluable | — | — | — | — |

Confusion matrices use true rows / predicted columns, absent then present:

- Hair: `[[368, 77], [160, 547]]`.
- Ruler: `[[1050, 39], [33, 30]]`.

Ruler presence is rare. Its high AUROC should be read alongside its lower
artifact-present F1; the selected threshold detected 30 of 63 rulers.

## Interpretation

Hair and ruler labels are linearly decodable from the model's decision features,
outperforming constant majority predictions. This shows the representation
retains artifact-related information. It does **not** establish that the lesion
classifier causally relies on those artifacts, or that LoRA introduced that
information; the optional pretrained-model control has not been run.

`diagnosis_by_artifact.csv` describes the lesion classifier's scores and errors
within each true diagnosis, recomputed during the frozen feature pass. For
example, annotated test nevi with rulers had a
mean melanoma score of 0.436 (43 images), versus 0.254 without rulers (980 images).
Their false-positive rates were 34.9% and 21.9%, respectively. These are
observational associations with potential acquisition/lesion confounding, not
paired artifact interventions. Scores are conditional on the A/B answer choices,
not calibrated disease probabilities.

## Saved outputs

- `results.json`: selected settings, full metrics, baselines, source hashes,
  feature/checkpoint provenance, and diagnosis-specific annotation coverage.
- `cohort.csv`: annotated images, original lesion groups/splits, diagnoses, and
  binary artifact labels.
- `hair_test_predictions.csv`, `ruler_test_predictions.csv`: individual test
  labels, scores, and predictions.
- `diagnosis_by_artifact.csv`: group counts and lesion-classification associations.
- `run_info.json`: runtime details. The Colab T4 was released after download.

Ignored side products remain local: `data/features/smolvlm_lora.npz` contains
the frozen features and lesion scores for all 7,818 original images;
`artifacts/smolvlm_artifact_probes/` contains the two readout weights/scalers.
The job bundle, output archive, and execution record are also in ignored
`artifacts/`. The original lesion-model weights and results were not overwritten.

Reproduce with the three commands in the root README's artifact-probe section.
