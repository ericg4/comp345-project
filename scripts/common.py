"""Paths, split loading, and metrics shared by every experiment arm."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
HAM_DIR = ROOT / "data" / "ham10000" / "skin-cancer-mnist-ham10000"
FEATURES_DIR = ROOT / "data" / "features"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"
SPLIT_CSV = ROOT / "splits" / "ham10000_nv_vs_mel.csv"
CLASSES = ["nv", "mel"]  # label 0 and label 1


def load_split(data_dir=HAM_DIR, split_csv=SPLIT_CSV):
    """The shared split (image_id, lesion_id, dx, label, split) plus each image's path."""
    split = pd.read_csv(split_csv)
    paths = {p.stem: p for p in Path(data_dir).glob("HAM10000_images_part_*/*.jpg")}
    split["path"] = split["image_id"].map(paths)
    if split["path"].isna().any():
        raise FileNotFoundError(f"HAM10000 images missing under {data_dir}; see README.md")
    return split


def metrics(labels, probs, threshold=0.5):
    """Metrics from integer labels (N,) and class probabilities (N, 2).

    Predicts melanoma when its probability is >= threshold (choose it on val).
    """
    preds = (probs[:, 1] >= threshold).astype(int)
    f1 = f1_score(labels, preds, average=None, labels=[0, 1])
    return {
        "macro_f1": float(f1.mean()),
        "f1": dict(zip(CLASSES, f1.tolist())),
        "auroc": float(roc_auc_score(labels, probs[:, 1])),
        # Rows are true classes, columns predicted, both in CLASSES order.
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    }
