#!/usr/bin/env python3
"""Run the frozen PanDerm ViT-L/16 encoder over every split image and save its features.

Saves data/features/panderm.npz (or panderm_random.npz with --random-init, a
no-pretraining control): `features` (N, 1024) in split-CSV row order, plus `image_ids`.
Preprocessing and the feature (final-norm CLS token) match PanDerm's own linear probe.
"""

import argparse
import importlib.util
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from common import FEATURES_DIR, ROOT, load_split

CHECKPOINT = ROOT / "data" / "panderm" / "panderm_ll_data6_checkpoint-499.pth"
MODEL_CODE = ROOT / "third_party" / "PanDerm" / "classification" / "models" / "modeling_finetune.py"
# PanDerm's constants, including its 0.228 where ImageNet uses 0.229.
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
])


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return TRANSFORM(Image.open(self.paths[i]).convert("RGB"))


def load_panderm(random_init=False):
    # Load PanDerm's model file on its own: its package __init__ needs open_clip.
    spec = importlib.util.spec_from_file_location("panderm_model", MODEL_CODE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["panderm_model"] = module  # timm's @register_model looks it up here.
    spec.loader.exec_module(module)

    model = module.panderm_large_patch16_224()
    model.head = torch.nn.Identity()  # Its constructor can't build without a head.
    if not random_init:
        # The checkpoint also holds the pretraining decoder; keep only the encoder.
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        encoder = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
        model.load_state_dict(encoder, strict=True)
    return model.eval()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-init", action="store_true", help="Skip pretrained weights (control)")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_panderm(args.random_init).to(device)
    split = load_split()
    loader = DataLoader(Images(split["path"].tolist()), batch_size=64, num_workers=4)
    features = [model.forward_features(batch.to(device), is_train=False).cpu() for batch in tqdm(loader)]
    features = torch.cat(features).numpy()

    out = FEATURES_DIR / ("panderm_random.npz" if args.random_init else "panderm.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=features, image_ids=split["image_id"].to_numpy(dtype=str))
    print(f"Saved {features.shape} to {out}")


if __name__ == "__main__":
    main()
