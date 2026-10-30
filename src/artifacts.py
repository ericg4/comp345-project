"""Synthetic ink and ruler overlays used by the insertion and ink-shortcut experiments.

Each artifact has three designs. paired_overlays() also returns a matched control:
the same RGBA pixels shuffled, so color, opacity, and area match but shape is lost.
"""

import hashlib

import numpy as np
from PIL import Image, ImageDraw

SEED = 42


def seed_for(*parts):
    """A deterministic 64-bit seed from any values."""
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


def overlay(artifact, design):
    """A transparent RGBA overlay: "ruler" (144x40) or "ink" (112x56), design 0-2."""
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
        return image
    # Draw at 3x and downsample for smooth, partially transparent marker edges.
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
    return image.resize((112, 56), Image.Resampling.LANCZOS)


def paired_overlays(artifact, design, rotation):
    """(target overlay, pixel-shuffled control), both rotated by rotation degrees."""
    target = overlay(artifact, design).rotate(rotation, expand=True)
    rgba = np.array(target)
    rng = np.random.default_rng(seed_for(SEED, "control", artifact, design, rotation))
    control = rgba.reshape(-1, 4)[rng.permutation(rgba.shape[0] * rgba.shape[1])].reshape(rgba.shape)
    return target, Image.fromarray(control)


def composite(rgb, patch, x, y):
    """rgb (H, W, 3) array with the RGBA patch pasted at (x, y)."""
    image = Image.fromarray(rgb).convert("RGBA")
    image.alpha_composite(patch, (int(x), int(y)))
    return np.array(image.convert("RGB"))
