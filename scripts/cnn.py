#!/usr/bin/env python3
"""Train a small CNN from scratch on nevus vs. melanoma and save its embeddings.

    uv run python scripts/cnn.py
    uv run python scripts/linear_probe.py data/features/cnn.npz

Four conv blocks, global average pooling to a 256-d embedding, then a two-layer
head. Mini-batch SGD (batch 64) on class-weighted cross-entropy (NLL). The epoch
and melanoma decision threshold are chosen by validation macro F1; test is scored
once. Writes results/cnn_trained.json, per-image test predictions, the best
weights to data/cnn/cnn.pt, and the 256-d embeddings of every split image to
data/features/cnn.npz (split-CSV row order) for linear probing.
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import CLASSES, FEATURES_DIR, RESULTS_DIR, ROOT, load_split, metrics
from linear_probe import best_threshold

CHECKPOINT = ROOT / "data" / "cnn" / "cnn.pt"
BATCH_SIZE = 64
NORMALIZE = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# Same 224 crop as PanDerm. Training adds random crops and flips; no color jitter,
# so color artifacts (ink, the red box) stay as available as they are in the data.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    NORMALIZE,
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    NORMALIZE,
])


class Images(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def conv_block(c_in, c_out):
    """Two 3x3 conv-BN-ReLU layers, then 2x2 max pooling (halves height and width)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), hidden=128, dropout=0.3):
        super().__init__()
        widths = (3, *channels)
        self.features = nn.Sequential(*[conv_block(a, b) for a, b in zip(widths, widths[1:])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(channels[-1], hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(CLASSES)),
        )

    def embed(self, x):
        """The pooled conv features (N, 256): what the linear probe reads."""
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.head(self.embed(x))


@torch.no_grad()
def predict(model, loader, device):
    """Class probabilities (N, 2) and embeddings (N, 256), in loader order."""
    model.eval()
    probs, embeddings = [], []
    for x, _ in loader:
        z = model.embed(x.to(device))
        probs.append(model.head(z).softmax(1).cpu())
        embeddings.append(z.cpu())
    return torch.cat(probs).numpy(), torch.cat(embeddings).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    split = load_split()
    paths, labels = split["path"].tolist(), split["label"].to_numpy()

    def loader(rows, transform, shuffle=False):
        rows = np.flatnonzero(rows)
        images = Images([paths[i] for i in rows], labels[rows], transform)
        return DataLoader(images, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=args.workers,
                          persistent_workers=args.workers > 0)

    train_loader = loader(split["split"] == "train", TRAIN_TRANSFORM, shuffle=True)
    val_rows = (split["split"] == "val").to_numpy()
    val_loader = loader(val_rows, EVAL_TRANSFORM)
    val_y = labels[val_rows]

    model = SmallCNN().to(device)
    print(f"SmallCNN: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # Weight each class by inverse frequency so melanoma counts as much as nevus.
    train_y = torch.from_numpy(labels[(split["split"] == "train").to_numpy()])
    loss_fn = nn.CrossEntropyLoss(weight=(len(train_y) / (len(CLASSES) * torch.bincount(train_y).float())).to(device))

    best = {"val_macro_f1": -1.0}
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        scheduler.step()
        threshold, score = best_threshold(val_y, predict(model, val_loader, device)[0])
        history.append({"epoch": epoch, "train_loss": total / len(train_y), "val_macro_f1": score})
        print(f"epoch {epoch}: train loss {total / len(train_y):.4f}, val macro F1 {score:.4f} "
              f"(threshold {threshold:.2f})")
        if score > best["val_macro_f1"]:
            best = {"val_macro_f1": score, "epoch": epoch, "threshold": threshold}
            CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    probs, embeddings = predict(model, loader(np.ones(len(split), bool), EVAL_TRANSFORM), device)

    out = FEATURES_DIR / "cnn.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=embeddings, image_ids=split["image_id"].to_numpy(dtype=str))

    test_rows = (split["split"] == "test").to_numpy()
    test_probs = probs[test_rows]
    test = metrics(labels[test_rows], test_probs, best["threshold"])
    RESULTS_DIR.mkdir(exist_ok=True)
    result = {"model": "cnn", "args": vars(args), "chosen": best, "test": test, "history": history}
    (RESULTS_DIR / "cnn_trained.json").write_text(json.dumps(result, indent=2) + "\n")
    predictions = split[test_rows][["image_id", "lesion_id", "label"]]
    predictions.assign(prob_mel=test_probs[:, 1], pred=(test_probs[:, 1] >= best["threshold"]).astype(int)).to_csv(
        RESULTS_DIR / "cnn_trained_test_predictions.csv", index=False)

    print(f"\nChose epoch {best['epoch']}, threshold {best['threshold']:.2f}; saved {embeddings.shape} to {out}; "
          f"test: {json.dumps(test)}")


if __name__ == "__main__":
    main()
