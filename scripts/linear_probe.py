#!/usr/bin/env python3
"""Train a linear head on saved features and evaluate it on the test split.

    uv run python scripts/linear_probe.py data/features/panderm.npz

Mini-batch SGD (batch 64) on class-weighted cross-entropy (NLL). The learning
rate, weight decay, epoch, and melanoma decision threshold are chosen by
validation macro F1; test is scored once. Writes results/<features name>.json
and per-image test predictions.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import CLASSES, RESULTS_DIR, load_split, metrics

LRS = [1e-3, 1e-2, 1e-1]
# Strong values matter: 1024 features vs. ~800 training melanomas overfits easily.
WEIGHT_DECAYS = [1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
THRESHOLDS = np.linspace(0.05, 0.95, 91)
EPOCHS = 100
BATCH_SIZE = 64


def best_threshold(labels, probs):
    """The melanoma threshold with the highest macro F1, and that F1."""
    scores = [metrics(labels, probs, t)["macro_f1"] for t in THRESHOLDS]
    return float(THRESHOLDS[np.argmax(scores)]), max(scores)


def train(x, y, val_x, val_y, lr, weight_decay):
    """Return (val macro F1, epoch, threshold, weights) at the best validation epoch."""
    torch.manual_seed(0)
    head = nn.Linear(x.shape[1], len(CLASSES))
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # Weight each class by inverse frequency so melanoma counts as much as nevus.
    loss_fn = nn.CrossEntropyLoss(weight=len(y) / (len(CLASSES) * torch.bincount(y).float()))
    best = (-1.0, -1, 0.5, None)
    for epoch in range(EPOCHS):
        for rows in torch.randperm(len(x)).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss_fn(head(x[rows]), y[rows]).backward()
            optimizer.step()
        with torch.no_grad():
            threshold, score = best_threshold(val_y.numpy(), head(val_x).softmax(1).numpy())
        if score > best[0]:
            best = (score, epoch, threshold, {k: v.clone() for k, v in head.state_dict().items()})
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", type=Path, help=".npz with `features` and `image_ids`")
    args = parser.parse_args()

    split = load_split()
    saved = np.load(args.features)
    assert (saved["image_ids"] == split["image_id"].to_numpy()).all(), "features not in split order"
    x_all = torch.from_numpy(saved["features"])
    data = {}
    for name in ("train", "val", "test"):
        rows = (split["split"] == name).to_numpy()
        data[name] = (x_all[rows], torch.from_numpy(split.loc[rows, "label"].to_numpy()))
    # Standardize with training-set statistics.
    mean, std = data["train"][0].mean(0), data["train"][0].std(0) + 1e-6
    data = {name: ((x - mean) / std, y) for name, (x, y) in data.items()}

    runs = []
    for lr, weight_decay in itertools.product(LRS, WEIGHT_DECAYS):
        score, epoch, threshold, weights = train(*data["train"], *data["val"], lr, weight_decay)
        runs.append({"lr": lr, "weight_decay": weight_decay, "val_macro_f1": score, "epoch": epoch,
                     "threshold": threshold, "weights": weights})
        print(f"lr={lr:g} wd={weight_decay:g}: val macro F1 {score:.4f} (epoch {epoch}, threshold {threshold:.2f})")
    best = max(runs, key=lambda run: run["val_macro_f1"])

    head = nn.Linear(x_all.shape[1], len(CLASSES))
    head.load_state_dict(best.pop("weights"))
    test_x, test_y = data["test"]
    with torch.no_grad():
        test_probs = head(test_x).softmax(1).numpy()
    test = metrics(test_y.numpy(), test_probs, best["threshold"])

    name = args.features.stem
    RESULTS_DIR.mkdir(exist_ok=True)
    result = {"features": name, "chosen": best, "test": test,
              "grid": [{k: v for k, v in run.items() if k != "weights"} for run in runs]}
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    predictions = split[split["split"] == "test"][["image_id", "lesion_id", "label"]]
    predictions.assign(prob_mel=test_probs[:, 1], pred=(test_probs[:, 1] >= best["threshold"]).astype(int)).to_csv(
        RESULTS_DIR / f"{name}_test_predictions.csv", index=False)

    print(f"\nChosen lr={best['lr']:g} wd={best['weight_decay']:g} threshold={best['threshold']:.2f}; "
          f"test: {json.dumps(test)}")


if __name__ == "__main__":
    main()
