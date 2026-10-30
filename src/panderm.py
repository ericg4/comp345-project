"""Run the frozen PanDerm ViT-L/16 encoder over every split image and save its features.

    uv run python src/panderm.py [--random-init]

Writes data/features/panderm/pretrained.npz (or random_init.npz, a no-pretraining
control). Preprocessing and the feature (final-norm CLS token, 1024-d) match
PanDerm's own linear probe.
"""

import argparse
import importlib.util
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from common import DATA_DIR, FEATURES_DIR, ROOT, Images, load_split, pick_device

CHECKPOINT = DATA_DIR / "panderm" / "panderm_ll_data6_checkpoint-499.pth"
MODEL_CODE = ROOT / "third_party" / "PanDerm" / "classification" / "models" / "modeling_finetune.py"
# PanDerm's constants, including its 0.228 where ImageNet uses 0.229.
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225)),
])


def load_panderm(random_init=False):
    # Import PanDerm's model file alone: its package __init__ needs open_clip.
    spec = importlib.util.spec_from_file_location("panderm_model", MODEL_CODE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["panderm_model"] = module  # timm's @register_model looks it up here.
    spec.loader.exec_module(module)

    model = module.panderm_large_patch16_224()
    model.head = torch.nn.Identity()
    if not random_init:
        # The checkpoint also holds the pretraining decoder; keep only the encoder.
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict({k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")})
    return model.eval()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--random-init", action="store_true", help="Skip pretrained weights (control)")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    model = load_panderm(args.random_init).to(device)
    split = load_split()
    loader = DataLoader(Images(split["path"].tolist(), split["label"].to_numpy(), TRANSFORM), batch_size=64,
                        num_workers=4)
    features = torch.cat([model.forward_features(x.to(device), is_train=False).cpu() for x, _ in tqdm(loader)])

    out = FEATURES_DIR / "panderm" / ("random_init.npz" if args.random_init else "pretrained.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=features.numpy(), image_ids=split["image_id"].to_numpy(dtype=str))
    print(f"Saved {tuple(features.shape)} to {out}")


if __name__ == "__main__":
    main()
