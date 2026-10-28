#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy==2.1.3", "Pillow==11.3.0"]
# ///
r"""Generate the synthetic ink/ruler graphics used in our insertion experiment.

This file is standalone: copy it alone. With uv, dependencies install in an
isolated environment without installing the surrounding project's packages.

Transparent ink asset:
    uv run scripts/synthetic_artifacts.py --output ink.png

Apply ink to an image at explicit native-image pixel coordinates:
    uv run scripts/synthetic_artifacts.py --image lesion.jpg --x 20 --y 20 \
        --design 1 --output with_ink.png --control-output ink_control.png

Ruler asset:
    uv run scripts/synthetic_artifacts.py --artifact ruler --output ruler.png

Without uv: install numpy==2.1.3 and Pillow==11.3.0, then run this file with Python.
There are no model, dataset, segmentation, or masking-code dependencies.
Placement is manual: choose coordinates that avoid the lesion. This script does
not identify or protect lesions. Outputs are lossless PNG files.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SEED = 42


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


def overlay(artifact, design):
    if artifact == "ruler":
        image = Image.new("RGBA", (144, 40), [(235, 235, 220, 190), (225, 235, 240, 180), (40, 45, 48, 190)][design])
        draw = ImageDraw.Draw(image)
        color = (15, 20, 20, 235) if design < 2 else (245, 245, 230, 240)
        spacing = [12, 10, 14][design]
        draw.line((0, 1, 143, 1), fill=color, width=2)
        for i, x in enumerate(range(6, 140, spacing)):
            draw.line((x, 2, x, 22 if i % 5 == 0 else 14 if i % 2 == 0 else 9), fill=color, width=2)
            if i % 5 == 0:
                draw.text((x + 3, 23), str(i // 5), fill=color)
    else:
        # Supersample strokes for smooth, partially transparent marker edges.
        image = Image.new("RGBA", (112 * 3, 56 * 3))
        draw = ImageDraw.Draw(image)
        color = [(74, 35, 128, 205), (38, 53, 142, 200), (101, 34, 116, 200)][design]
        if design == 0:
            draw.arc((15, 12, 315, 156), 170, 345, fill=color, width=12)
        elif design == 1:
            draw.line([(18, 120), (90, 48), (195, 36), (315, 96)], fill=color, width=12, joint="curve")
        else:
            draw.arc((27, 9, 300, 159), 5, 175, fill=color, width=15)
            draw.line((270, 126, 315, 150), fill=color, width=9)
        image = image.resize((112, 56), Image.Resampling.LANCZOS)
    return image


def paired_overlays(artifact, design, rotation):
    target = overlay(artifact, design).rotate(rotation, expand=True)
    rgba = np.array(target)
    rng = np.random.default_rng(seed_for(SEED, "control", artifact, design, rotation))
    # Permute whole RGBA tuples: identical color/alpha histogram and painted
    # pixel count, at the same location. This deliberately destroys structure.
    control = rgba.reshape(-1, 4)[rng.permutation(rgba.shape[0] * rgba.shape[1])].reshape(rgba.shape)
    assert np.array_equal(np.sort(rgba.reshape(-1, 4), axis=0), np.sort(control.reshape(-1, 4), axis=0))
    return target, Image.fromarray(control)


def composite(rgb, patch, x, y):
    image = Image.fromarray(rgb).convert("RGBA")
    image.alpha_composite(patch, (int(x), int(y)))
    return np.array(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", choices=["ink", "ruler"], default="ink")
    parser.add_argument("--design", type=int, choices=range(3), default=0)
    parser.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    parser.add_argument("--image", type=Path, help="Optional source image; otherwise save a transparent overlay asset")
    parser.add_argument("--x", type=int, default=0, help="Left coordinate in source-image pixels")
    parser.add_argument("--y", type=int, default=0, help="Top coordinate in source-image pixels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-output", type=Path, help="Optional matched RGBA pixel-permutation control")
    args = parser.parse_args()
    outputs = [args.output] + ([args.control_output] if args.control_output else [])
    if len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("Target and control outputs must be separate files.")
    for path in outputs:
        if path.suffix.lower() != ".png":
            parser.error("Use .png outputs to preserve transparency and avoid JPEG recompression.")
        if path.exists():
            parser.error(f"Output already exists: {path}")
    target, control = paired_overlays(args.artifact, args.design, args.rotation)
    pictures = [target] + ([control] if args.control_output else [])
    if args.image:
        with Image.open(args.image) as image:
            rgb = np.array(image.convert("RGB"))
        width, height = target.size
        if args.x < 0 or args.y < 0 or args.x + width > rgb.shape[1] or args.y + height > rgb.shape[0]:
            parser.error(f"The {width}x{height} overlay must fit completely inside the source image.")
        pictures = [Image.fromarray(composite(rgb, patch, args.x, args.y)) for patch in pictures]
    for path, image in zip(outputs, pictures):
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
