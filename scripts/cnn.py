#!/usr/bin/env python3
"""Train a small CNN from scratch on nevus vs. melanoma and save its embeddings.

    uv run python scripts/cnn.py --size 28
    uv run python scripts/linear_probe.py data/features/cnn/28.npz

Up to four conv blocks (four at 224px, three at 28px), global average pooling to
an embedding (256-d at 224px, 128-d at 28px), then a two-layer head. Mini-batch
SGD (batch 64) on class-weighted cross-entropy (NLL). The epoch and melanoma
decision threshold are chosen by validation macro F1; test is scored once.
Writes results/cnn/<size>/trained.json and per-image test predictions, the best
weights to data/cnn/<size>.pt, and the embeddings of every split image to
data/features/cnn/<size>.npz (split-CSV row order) for linear probing, whose
results then land next to these in results/cnn/<size>/.
"""

import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from common import CLASSES, FEATURES_DIR, RESULTS_DIR, ROOT, load_split, metrics
from linear_probe import best_threshold

BATCH_SIZE = 64
CHANNELS = (32, 64, 128, 256)
# At or below this size, every image is resized once up front and kept in memory.
CACHE_MAX_SIZE = 64
NORMALIZE = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def transforms_for(size):
    """(resize, train, eval) transforms for size x size inputs.

    Resize the short side to 8/7 of size, then crop size x size: at 224 this is the
    same 256 -> 224 crop as PanDerm, at 28 it is 32 -> 28. Training adds random
    crops and flips; no color jitter, so color artifacts (ink, the red box) stay as
    available as they are in the data.
    """
    resize = transforms.Resize(round(size * 256 / 224))
    train = transforms.Compose([
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    evaluate = transforms.Compose([transforms.CenterCrop(size), transforms.ToTensor(), NORMALIZE])
    return resize, train, evaluate


class Images(Dataset):
    """Images given as paths (loaded and resized per access) or as resized PIL images."""

    def __init__(self, images, labels, resize, transform):
        self.images, self.labels, self.resize, self.transform = images, labels, resize, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        if not isinstance(image, Image.Image):
            image = self.resize(Image.open(image).convert("RGB"))
        return self.transform(image), self.labels[i]


def conv_block(c_in, c_out):
    """Two 3x3 conv-BN-ReLU layers, then 2x2 max pooling (halves height and width)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


def depth(size):
    """Conv blocks for a size: up to four, keeping the last feature map at least 3x3."""
    blocks = 0
    while blocks < len(CHANNELS) and size // 2 >= 3:
        size //= 2
        blocks += 1
    return blocks


class SmallCNN(nn.Module):
    def __init__(self, channels=CHANNELS, hidden=128, dropout=0.3):
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
        """The pooled conv features (N, channels[-1]): what the linear probe reads."""
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.head(self.embed(x))


@torch.no_grad()
def predict(model, loader, device):
    """Class probabilities (N, 2), embeddings (N, channels[-1]), and melanoma-minus-nevus
    logit margins (N,), in loader order."""
    model.eval()
    probs, embeddings, margins = [], [], []
    for x, _ in loader:
        z = model.embed(x.to(device))
        logits = model.head(z)
        probs.append(logits.softmax(1).cpu())
        embeddings.append(z.cpu())
        margins.append((logits[:, 1] - logits[:, 0]).cpu())
    return torch.cat(probs).numpy(), torch.cat(embeddings).numpy(), torch.cat(margins).numpy()


def pick_device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_images(paths, size, workers):
    """The images to train on and the DataLoader worker count to use.

    Paths, or at sizes up to CACHE_MAX_SIZE, resized PIL images held in memory
    (decoding the full JPEGs every epoch would otherwise dominate).
    """
    if size > CACHE_MAX_SIZE:
        return paths, workers
    resize = transforms_for(size)[0]
    return [resize(Image.open(path).convert("RGB")) for path in tqdm(paths, desc="Caching resized images")], 0


def make_loader(images, labels, size, transform, workers=0, shuffle=False):
    dataset = Images(images, labels, transforms_for(size)[0], transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0)


def train(images, labels, parts, size, checkpoint, epochs=40, lr=0.01, weight_decay=5e-4, workers=0, seed=0,
          verbose=True):
    """Train on the rows where parts == "train", choosing epoch and threshold on parts == "val".

    images and labels cover every row (paths or PIL images from load_images). Saves
    the best epoch's weights to checkpoint; returns (model with them loaded, best, history).
    """
    torch.manual_seed(seed)
    device = pick_device()
    _, train_transform, eval_transform = transforms_for(size)
    train_rows, val_rows = np.flatnonzero(parts == "train"), np.flatnonzero(parts == "val")
    train_loader = make_loader([images[i] for i in train_rows], labels[train_rows], size, train_transform, workers,
                               shuffle=True)
    val_loader = make_loader([images[i] for i in val_rows], labels[val_rows], size, eval_transform, workers)
    val_y = labels[val_rows]

    model = SmallCNN(CHANNELS[:depth(size)]).to(device)
    if verbose:
        print(f"SmallCNN at {size}px: {len(model.features)} conv blocks, "
              f"{sum(p.numel() for p in model.parameters()):,} parameters on {device}")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    # Weight each class by inverse frequency so melanoma counts as much as nevus.
    train_y = torch.from_numpy(labels[train_rows])
    loss_fn = nn.CrossEntropyLoss(weight=(len(train_y) / (len(CLASSES) * torch.bincount(train_y).float())).to(device))

    best = {"val_macro_f1": -1.0}
    history = []
    for epoch in range(epochs):
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
        if verbose:
            print(f"epoch {epoch}: train loss {total / len(train_y):.4f}, val macro F1 {score:.4f} "
                  f"(threshold {threshold:.2f})")
        if score > best["val_macro_f1"]:
            best = {"val_macro_f1": score, "epoch": epoch, "threshold": threshold}
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    return model, best, history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4, help="Ignored when images are cached in memory")
    parser.add_argument("--size", type=int, default=224, help="Input height and width in pixels")
    args = parser.parse_args()

    split = load_split()
    labels = split["label"].to_numpy()
    images, workers = load_images(split["path"].tolist(), args.size, args.workers)
    model, best, history = train(images, labels, split["split"].to_numpy(), args.size,
                                 ROOT / "data" / "cnn" / f"{args.size}.pt", args.epochs, args.lr, args.weight_decay,
                                 workers)
    eval_transform = transforms_for(args.size)[2]
    probs, embeddings, _ = predict(model, make_loader(images, labels, args.size, eval_transform, workers),
                                   pick_device())

    out = FEATURES_DIR / "cnn" / f"{args.size}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=embeddings, image_ids=split["image_id"].to_numpy(dtype=str))

    test_rows = (split["split"] == "test").to_numpy()
    test_probs = probs[test_rows]
    test = metrics(labels[test_rows], test_probs, best["threshold"])
    results = RESULTS_DIR / "cnn" / str(args.size)
    results.mkdir(parents=True, exist_ok=True)
    result = {"model": "cnn", "args": vars(args), "chosen": best, "test": test, "history": history}
    (results / "trained.json").write_text(json.dumps(result, indent=2) + "\n")
    predictions = split[test_rows][["image_id", "lesion_id", "label"]]
    predictions.assign(prob_mel=test_probs[:, 1], pred=(test_probs[:, 1] >= best["threshold"]).astype(int)).to_csv(
        results / "trained_test_predictions.csv", index=False)

    print(f"\nChose epoch {best['epoch']}, threshold {best['threshold']:.2f}; saved {embeddings.shape} to {out}; "
          f"test: {json.dumps(test)}")


if __name__ == "__main__":
    main()
