"""The planted ink shortcut (see ink_shortcut.py) for a linear probe on frozen PanDerm features.

    uv run python src/panderm_ink_shortcut.py

Same eligible images, ink stamps, levels, and test conditions as the CNN
experiment. The encoder is frozen, so only the linear head sees the ink bias: the
shortcut can appear only if ink is linearly readable in PanDerm's features. Clean
features come from data/features/panderm/pretrained.npz; inked ones are extracted
once to data/features/panderm/inked.npz. A probe trained without ink is the
baseline. Writes results/panderm/pretrained/ink_shortcut/.
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from artifacts import composite
from common import CLASSES, FEATURES_DIR, RESULTS_DIR, load_split, metrics, pick_device
from ink_shortcut import LEVELS, SEED, STAMPS, cluster_ci, has_ink, placements
from linear_probe import LRS, WEIGHT_DECAYS, train
from panderm import TRANSFORM, load_panderm

CLEAN = FEATURES_DIR / "panderm" / "pretrained.npz"
INKED = FEATURES_DIR / "panderm" / "inked.npz"


class InkedImages(Dataset):
    def __init__(self, table):
        self.rows = list(table[["path", "design", "rotation", "x", "y"]].itertuples(index=False, name=None))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, design, rotation, x, y = self.rows[i]
        rgb = np.array(Image.open(path).convert("RGB"))
        return TRANSFORM(Image.fromarray(composite(rgb, STAMPS[int(design), int(rotation)], x, y)))


@torch.no_grad()
def inked_features(eligible):
    """PanDerm features (N, 1024) of every eligible image with its ink stamp, cached to INKED."""
    image_ids = eligible["image_id"].to_numpy(dtype=str)
    if INKED.exists():
        saved = np.load(INKED)
        assert (saved["image_ids"] == image_ids).all(), f"{INKED} is stale; delete it"
        return saved["features"]
    device = pick_device()
    model = load_panderm().to(device)
    loader = DataLoader(InkedImages(eligible), batch_size=64, num_workers=4)
    features = torch.cat([model.forward_features(x.to(device), is_train=False).cpu()
                          for x in tqdm(loader, desc="PanDerm on inked images")]).numpy()
    np.savez(INKED, features=features, image_ids=image_ids)
    return features


def fit(x, labels, parts, seed):
    """The best linear head over the lr x weight-decay grid, on train-standardized features."""
    x = torch.from_numpy(x)
    y = torch.from_numpy(labels)
    train_rows, val_rows = parts == "train", parts == "val"
    mean, std = x[train_rows].mean(0), x[train_rows].std(0) + 1e-6
    x = (x - mean) / std
    best = None
    for lr, weight_decay in itertools.product(LRS, WEIGHT_DECAYS):
        score, epoch, threshold, weights = train(x[train_rows], y[train_rows], x[val_rows], y[val_rows], lr,
                                                 weight_decay, seed)
        if best is None or score > best["val_macro_f1"]:
            best = {"lr": lr, "weight_decay": weight_decay, "val_macro_f1": score, "epoch": epoch,
                    "threshold": threshold, "weights": weights}
    head = nn.Linear(x.shape[1], len(CLASSES))
    head.load_state_dict(best.pop("weights"))
    return head, mean, std, best


@torch.no_grad()
def evaluate(head, mean, std, threshold, test, clean, inked, conditions):
    """Per-condition test metrics, plus the paired all-minus-clean logit-margin effects."""
    labels = test["label"].to_numpy()
    outputs, scores = {}, {}
    for name, ink in conditions.items():
        logits = head((torch.from_numpy(np.where(ink[:, None], inked, clean)) - mean) / std)
        probs = logits.softmax(1).numpy()
        outputs[name] = (probs, (logits[:, 1] - logits[:, 0]).numpy())
        scores[name] = metrics(labels, probs, threshold)
        scores[name].update(mel_ink_rate=float(ink[labels == 1].mean()), nv_ink_rate=float(ink[labels == 0].mean()))
    clean_preds = outputs["clean"][0][:, 1] >= threshold
    for name, (probs, _) in outputs.items():
        scores[name]["flip_rate_vs_clean"] = float(((probs[:, 1] >= threshold) != clean_preds).mean())
    rng = np.random.default_rng(SEED)
    delta = outputs["all"][1] - outputs["clean"][1]
    effects = {}
    for diagnosis, rows in (("all", labels >= 0), ("nv", labels == 0), ("mel", labels == 1)):
        mean_shift, low, high = cluster_ci(delta[rows], test["lesion_id"].to_numpy()[rows], rng)
        effects[diagnosis] = {"n_images": int(rows.sum()), "all_minus_clean_margin": mean_shift, "ci95": [low, high]}
    return scores, effects


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    out = RESULTS_DIR / "panderm" / "pretrained" / "ink_shortcut"
    out.mkdir(parents=True, exist_ok=True)

    split = load_split()
    saved = np.load(CLEAN)
    assert (saved["image_ids"] == split["image_id"].to_numpy()).all(), "features not in split order"
    table = placements(split)  # Same row order as split.
    keep = (table["excluded"] == "").to_numpy()
    eligible = table[keep].reset_index(drop=True)
    clean_all, inked_all = saved["features"][keep], inked_features(eligible)
    labels, parts = eligible["label"].to_numpy(), eligible["split"].to_numpy()
    test_rows = parts == "test"
    test = eligible[test_rows]
    clean, inked = clean_all[test_rows], inked_all[test_rows]
    none, every = np.zeros(len(test), bool), np.ones(len(test), bool)

    summary, runs = [], []
    for level in [None, *LEVELS]:
        mel_share, nv_share = level or (0.0, 0.0)
        name = "baseline" if level is None else f"{round(mel_share * 100)}/{round(nv_share * 100)}"
        ink = has_ink(eligible, mel_share, nv_share)
        features = np.where(ink[:, None], inked_all, clean_all)
        conditions = ({"clean": none, "all": every} if level is None else
                      {"biased": has_ink(test, mel_share, nv_share), "clean": none, "all": every,
                       "flipped": has_ink(test, nv_share, mel_share)})
        for seed in [0] if level is None else args.seeds:
            head, mean, std, best = fit(features, labels, parts, seed)
            scores, effects = evaluate(head, mean, std, best["threshold"], test, clean, inked, conditions)
            runs.append({"model": name, "seed": seed, "chosen": best, "test": scores, "effects": effects})
            for condition, score in scores.items():
                summary.append({"model": name, "seed": seed, "condition": condition, "n": len(test),
                                **{k: score[k] for k in ("mel_ink_rate", "nv_ink_rate", "macro_f1", "auroc",
                                                         "sensitivity", "specificity", "flip_rate_vs_clean")},
                                "mel_f1": score["f1"]["mel"]})
            print(f"{name} seed {seed}: val F1 {best['val_macro_f1']:.3f}; test macro F1 "
                  + ", ".join(f"{c} {s['macro_f1']:.3f}" for c, s in scores.items())
                  + f"; ink margin shift mel {effects['mel']['all_minus_clean_margin']:+.2f}, "
                    f"nv {effects['nv']['all_minus_clean_margin']:+.2f}", flush=True)

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    protocol = {"seed": SEED, "levels": LEVELS, "seeds": args.seeds, "n_eligible": len(eligible)}
    (out / "results.json").write_text(json.dumps({"protocol": protocol, "runs": runs}, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
