# SmolVLM-500M LoRA baseline

Colab T4 run of `src/smolvlm.py train`: 5 epochs, 512×512 images, LoRA rank 8 on
the language model's q/v projections, frozen vocabulary head (A = nevus,
B = melanoma). Epoch 5 was chosen by validation average precision; test is
scored at threshold 0.5.

**This run used its own lesion-grouped split (`split.csv`: 5,440 / 1,189 / 1,189),
not `splits/ham10000_nv_vs_mel.csv`.** Compare it with the other models loosely.

| Macro-F1 | AUROC | Avg. precision | Accuracy | Melanoma recall | Nevus specificity |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.691 | 0.874 | 0.560 | 0.785 | 0.837 | 0.776 |

Test confusion matrix (rows true, columns predicted; nevus, melanoma):
`[[794, 229], [27, 139]]`. The weights are not in the repo.
