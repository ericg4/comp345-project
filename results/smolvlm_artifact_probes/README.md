# Artifact probes on the frozen SmolVLM classifier

Can hair and rulers be read linearly from the LoRA model's decision feature (the
960-d final prompt token before the vocabulary head)? Each artifact gets an L2
logistic regression; C and threshold are chosen on validation macro-F1. Labels
come from [DermArtifactDB](https://zenodo.org/records/18324962) (7,597 of 7,818
images annotated; the 221 missing are all melanomas and were excluded). Ink was
skipped: the test split has no ink-positive labels.

| Artifact | Test positives / negatives | Macro-F1 | Artifact F1 | AUROC | Majority macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hair | 707 / 445 | 0.789 | 0.822 | 0.872 | 0.380 |
| Ruler | 63 / 1,089 | 0.711 | 0.455 | 0.926 | 0.486 |

Both artifacts are decodable well above chance, so the representation carries
artifact information. That alone does not show the classifier *uses* it.
Observationally, test nevi with a ruler got a mean melanoma score of 0.436 (n=43)
vs. 0.254 without (n=980), with false-positive rates of 34.9% vs. 21.9%
(`diagnosis_by_artifact.csv`).

The script for this study is not in the repo.
