# Original Colab LoRA run

Completed on a Tesla T4: five epochs, 512×512 images, rank 8, learning rate 1e-4,
microbatch 8, accumulation 2, seed 42. Only language query/value LoRA adapters
were trained; the original vocabulary head was frozen. Epoch five was selected
by validation average precision. This run has not been retrained or overwritten.

This run used its own lesion-grouped split: 5,440 train, 1,189 validation, and
1,189 test images. **It differs from `splits/ham10000_nv_vs_mel.csv`.** Future
runs of `scripts/train_smolvlm.py` use the project's shared split by default.
Do not compare these historical scores directly with the current PanDerm run.

| Test metric, threshold 0.5 | Value |
| --- | ---: |
| Macro-F1 | 0.690885 |
| ROC-AUC | 0.873641 |
| Average precision | 0.559659 |
| Balanced accuracy | 0.806749 |
| Accuracy | 0.784693 |
| Melanoma recall | 0.837349 |

Test confusion matrix (true rows / predicted columns, nevus then melanoma):
`[[794, 229], [27, 139]]`.

The additional threshold analysis chose 0.585101 using validation balanced
accuracy only. Test macro-F1 was 0.695661 and melanoma recall was 0.753012.
The original 0.5-threshold metrics remain saved separately.

Files in this directory are the useful results to commit:

- `results_summary.json`: the complete experiment summary.
- `test_metrics.json`, `test_predictions.csv`: held-out scores and per-image results.
- `history.json`, `best_validation.json`, `validation_predictions.csv`: validation evidence.
- `threshold_analysis.json`: the additional validation-selected threshold analysis.
- `config.json`, `run_info.json`: settings, software versions, and runtime details.
- `split.csv`: the original image/lesion split without machine-specific paths;
  its validation split is named `val`, matching the project's split format.

Local side products are stored separately in the gitignored
[`artifacts/ham10000_lora/`](../../artifacts/ham10000_lora/):

- `best_adapter/`, `processor/`, `config.json`: the original model files for inference.
- `train.csv`, `validation.csv`, `test.csv`: untouched original split exports.
- `results.zip`: the original downloaded archive, unchanged.
- `provenance.zip`: the original executed notebook, scripts, and logs.
- `original_checksums.json`: SHA-256 hashes of every original run file.

Moving these files did not change their contents. The archives also preserve
the original layout and execution records. Ignored files remain available
locally; cloning the repository alone will not restore the trained weights.

Validation average precision improved from 0.496 to 0.602 across the five
epochs. A plateau was not established; validation macro-F1 peaked in epoch
three, while average precision selected epoch five.
