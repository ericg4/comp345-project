"""Paths, the shared split, metrics, and small helpers used by every model."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
HAM_DIR = DATA_DIR / "ham10000" / "skin-cancer-mnist-ham10000"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = ROOT / "results"
SPLIT_CSV = ROOT / "splits" / "ham10000_nv_vs_mel.csv"
CLASSES = ["nv", "mel"]  # label 0, label 1


def load_split(data_dir=HAM_DIR):
    """The shared split (image_id, lesion_id, dx, label, split) plus each image's path."""
    split = pd.read_csv(SPLIT_CSV)
    paths = {p.stem: p for p in Path(data_dir).glob("HAM10000_images_part_*/*.jpg")}
    split["path"] = split["image_id"].map(paths)
    if split["path"].isna().any():
        raise FileNotFoundError(f"HAM10000 images missing under {data_dir}; see README.md")
    return split


def metrics(labels, probs, threshold=0.5):
    """Metrics from integer labels (N,) and class probabilities (N, 2); melanoma if prob >= threshold."""
    preds = (probs[:, 1] >= threshold).astype(int)
    f1 = f1_score(labels, preds, average=None, labels=[0, 1])
    # Rows are true classes, columns predicted, both in CLASSES order.
    (tn, fp), (fn, tp) = confusion_matrix(labels, preds, labels=[0, 1])
    return {
        "macro_f1": float(f1.mean()),
        "f1": dict(zip(CLASSES, f1.tolist())),
        "auroc": float(roc_auc_score(labels, probs[:, 1])),
        "sensitivity": float(tp / (tp + fn)),  # melanoma recall
        "specificity": float(tn / (tn + fp)),  # nevus recall
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Images(Dataset):
    """(transformed image, label) pairs from paths or already-loaded PIL images."""

    def __init__(self, images, labels, transform, resize=None):
        self.images, self.labels, self.transform, self.resize = images, labels, transform, resize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
            if self.resize:
                image = self.resize(image)
        return self.transform(image), self.labels[i]
