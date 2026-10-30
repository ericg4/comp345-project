"""Train the 28px CNN with synthetic ink planted as a shortcut, then test with it removed.

    uv run python src/download.py --ham10000-segmentations --extract
    uv run python src/ink_shortcut.py

Each eligible image gets one fixed ink stamp (design, rotation, position) at least
15 native pixels from every lesion-mask pixel and inside the CNN's center crop.
At each level (melanoma share, nevus share), exactly that share of each class in
train and val carries its stamp. Each level trains the CNN with several seeds,
then scores test with ink as in training (biased), on no image (clean), on every
image (all), and with the shares swapped (flipped). The ink-free data/cnn/<size>.pt
is scored on clean and all as a baseline. Writes results/cnn/<size>/ink_shortcut/.
"""

import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from artifacts import composite, paired_overlays, seed_for
from cnn import CACHE_MAX_SIZE, load_model, make_loader, predict, train, transforms_for
from common import DATA_DIR, RESULTS_DIR, load_split, metrics, pick_device

MASK_DIR = (DATA_DIR / "ham10000_segmentations" / "HAM10000_segmentations_lesion_tschandl"
            / "HAM10000_segmentations_lesion_tschandl")
SEED = 42
# Native pixels kept free of ink around every lesion pixel.
MARGIN = 15
ROTATIONS = (0, 90, 180, 270)
# Share of the short side the CNN's center crop keeps: 224/256, and 28/32 at 28px.
CROP_FRACTION = 224 / 256
# (melanoma share, nevus share) of train and val images carrying ink.
LEVELS = [(0.5, 0.5), (0.7, 0.3), (0.9, 0.1), (1.0, 0.0)]
BOOTSTRAP = 2000
STAMPS = {(design, rotation): paired_overlays("ink", design, rotation)[0]
          for design in range(3) for rotation in ROTATIONS}


def place(image_id):
    """This image's ink stamp as {design, rotation, x, y, width, height}, or the reason it has none."""
    mask = np.array(Image.open(MASK_DIR / f"{image_id}_segmentation.png")) > 127
    if not mask.any():
        return "empty_mask"
    # Foreground at a corner suggests reversed polarity or a lesion filling the frame.
    if mask[[0, 0, -1, -1], [0, -1, 0, -1]].any():
        return "foreground_at_corner"
    protected = distance_transform_edt(~mask) <= MARGIN
    height, width = mask.shape
    side = min(width, height) * CROP_FRACTION
    x0, x1 = int(np.ceil((width - side) / 2)), int((width + side) // 2)
    y0, y1 = int(np.ceil((height - side) / 2)), int((height + side) // 2)
    # Summed-area table of protected pixels in the visible region: a w x h stamp at
    # (x, y) is safe when its rectangle sums to zero.
    table = np.pad(protected[y0:y1, x0:x1].cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    rng = np.random.default_rng(seed_for(SEED, "ink_shortcut", image_id))
    choices = list(STAMPS)
    for k in rng.permutation(len(choices)):
        design, rotation = choices[k]
        w, h = STAMPS[design, rotation].size
        covered = table[h:, w:] - table[:-h, w:] - table[h:, :-w] + table[:-h, :-w]
        ys, xs = np.nonzero(covered == 0)
        if len(ys):
            i = rng.integers(len(ys))
            return {"design": design, "rotation": rotation, "x": x0 + int(xs[i]), "y": y0 + int(ys[i]),
                    "width": w, "height": h}
    return "no_safe_position"


def placements(split):
    """split plus each image's stamp (or exclusion reason) and its rank in the ink draw."""
    rows = []
    for image_id in tqdm(split["image_id"], desc="Placing ink"):
        stamp = place(image_id)
        rows.append(stamp if isinstance(stamp, dict) else {"excluded": stamp})
    table = pd.concat([split.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    table["excluded"] = table["excluded"].fillna("") if "excluded" in table else ""
    stamp = ["design", "rotation", "x", "y", "width", "height"]
    table[stamp] = table[stamp].astype("Int64")
    eligible = table["excluded"] == ""
    draw = [np.random.default_rng(seed_for(SEED, "ink_draw", i)).random() for i in table["image_id"]]
    table.loc[eligible, "draw"] = np.array(draw)[eligible]
    groups = table[eligible].groupby(["split", "label"])["draw"]
    table.loc[eligible, "draw_rank"] = groups.rank(method="first") - 1
    table.loc[eligible, "group_size"] = groups.transform("size")
    return table


def has_ink(table, mel_share, nv_share):
    """Which rows of an eligible-only table carry ink at these shares (exact counts per split and class)."""
    share = np.where(table["label"] == 1, mel_share, nv_share)
    return (table["draw_rank"] < np.round(share * table["group_size"])).to_numpy()


def cluster_ci(values, lesions, rng):
    """Mean and 95% whole-lesion bootstrap interval."""
    codes, lesion = np.unique(lesions, return_inverse=True)
    sums, counts = np.bincount(lesion, values), np.bincount(lesion)
    draws = rng.integers(len(codes), size=(BOOTSTRAP, len(codes)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(model, threshold, test, clean, inked, conditions, size):
    """Per-condition metrics and per-image outputs on the test rows, plus the paired all-minus-clean effects."""
    eval_transform = transforms_for(size)[2]
    labels = test["label"].to_numpy()
    outputs, scores = {}, {}
    for name, ink in conditions.items():
        images = [inked[i] if has else clean[i] for i, has in zip(test.index, ink)]
        probs, _, margins = predict(model, make_loader(images, labels, size, eval_transform), pick_device())
        outputs[name] = (probs, margins)
        score = metrics(labels, probs, threshold)
        score.update(mel_ink_rate=float(ink[labels == 1].mean()), nv_ink_rate=float(ink[labels == 0].mean()))
        scores[name] = score
    clean_preds = outputs["clean"][0][:, 1] >= threshold
    for name, (probs, _) in outputs.items():
        scores[name]["flip_rate_vs_clean"] = float(((probs[:, 1] >= threshold) != clean_preds).mean())
    rng = np.random.default_rng(SEED)
    delta = outputs["all"][1] - outputs["clean"][1]
    effects = {}
    for diagnosis, rows in (("all", labels >= 0), ("nv", labels == 0), ("mel", labels == 1)):
        mean, low, high = cluster_ci(delta[rows], test["lesion_id"].to_numpy()[rows], rng)
        effects[diagnosis] = {"n_images": int(rows.sum()), "all_minus_clean_margin": mean, "ci95": [low, high]}
    return scores, effects, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=28, help=f"Input size; at most {CACHE_MAX_SIZE} (images are cached)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if args.size > CACHE_MAX_SIZE:
        parser.error(f"--size above {CACHE_MAX_SIZE} would need clean and inked copies of every image in memory")
    out = RESULTS_DIR / "cnn" / str(args.size) / "ink_shortcut"
    out.mkdir(parents=True, exist_ok=True)

    table = placements(load_split())
    table.drop(columns="path").to_csv(out / "placements.csv", index=False)
    exclusions = table.groupby(["split", "dx"])["excluded"].value_counts().unstack(fill_value=0)
    print(f"Ink stamps placed; images by split, diagnosis, and exclusion ('' = eligible):\n{exclusions}\n")

    eligible = table[table["excluded"] == ""].reset_index(drop=True)
    resize = transforms_for(args.size)[0]
    clean, inked = [], []
    for row in tqdm(eligible.itertuples(), total=len(eligible), desc="Caching clean and inked images"):
        rgb = np.array(Image.open(row.path).convert("RGB"))
        clean.append(resize(Image.fromarray(rgb)))
        inked.append(resize(Image.fromarray(composite(rgb, STAMPS[row.design, row.rotation], row.x, row.y))))
    labels, parts = eligible["label"].to_numpy(), eligible["split"].to_numpy()
    test = eligible[parts == "test"]

    summary, runs = [], []

    def record(model_name, seed, level, scores, effects):
        for name, score in scores.items():
            summary.append({"model": model_name, "seed": seed, "condition": name, "n": len(test),
                            **{k: score[k] for k in ("mel_ink_rate", "nv_ink_rate", "macro_f1", "auroc",
                                                     "sensitivity", "specificity", "flip_rate_vs_clean")},
                            "mel_f1": score["f1"]["mel"]})
        runs.append({"model": model_name, "seed": seed, "level": level, "test": scores, "effects": effects})

    baseline = load_model(args.size, DATA_DIR / "cnn" / f"{args.size}.pt", pick_device())
    threshold = json.loads((RESULTS_DIR / "cnn" / str(args.size) / "trained.json").read_text())["chosen"]["threshold"]
    conditions = {"clean": np.zeros(len(test), bool), "all": np.ones(len(test), bool)}
    scores, effects, _ = evaluate(baseline, threshold, test, clean, inked, conditions, args.size)
    record("baseline", None, None, scores, effects)
    print(f"baseline (no ink in training): test macro F1 clean {scores['clean']['macro_f1']:.3f}, "
          f"all {scores['all']['macro_f1']:.3f}; ink margin shift mel {effects['mel']['all_minus_clean_margin']:+.2f}, "
          f"nv {effects['nv']['all_minus_clean_margin']:+.2f}")

    for mel_share, nv_share in LEVELS:
        level = f"{round(mel_share * 100)}/{round(nv_share * 100)}"
        ink = has_ink(eligible, mel_share, nv_share)
        images = [inked[i] if has else clean[i] for i, has in enumerate(ink)]
        test_ink = lambda m, n: has_ink(test, m, n)
        conditions = {"biased": test_ink(mel_share, nv_share), "clean": np.zeros(len(test), bool),
                      "all": np.ones(len(test), bool), "flipped": test_ink(nv_share, mel_share)}
        for seed in args.seeds:
            checkpoint = DATA_DIR / "cnn" / "ink_shortcut" / f"{args.size}_{level.replace('/', '-')}_seed{seed}.pt"
            model, best, history = train(images, labels, parts, args.size, checkpoint, epochs=args.epochs, seed=seed)
            scores, effects, _ = evaluate(model, best["threshold"], test, clean, inked, conditions, args.size)
            record(level, seed, level, scores, effects)
            runs[-1].update(chosen=best, history=history)
            print(f"{level} seed {seed}: val F1 {best['val_macro_f1']:.3f}; test macro F1 "
                  + ", ".join(f"{name} {s['macro_f1']:.3f}" for name, s in scores.items())
                  + f"; ink margin shift mel {effects['mel']['all_minus_clean_margin']:+.2f}, "
                    f"nv {effects['nv']['all_minus_clean_margin']:+.2f}")

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    counts = {f"{split}/{dx}/{reason or 'eligible'}": int(n) for (split, dx), row in exclusions.iterrows()
              for reason, n in row.items() if n}
    protocol = {"seed": SEED, "margin_native_pixels": MARGIN, "crop_fraction": CROP_FRACTION, "levels": LEVELS,
                "bootstrap": BOOTSTRAP, "size": args.size, "seeds": args.seeds, "epochs": args.epochs,
                "n_eligible": len(eligible), "counts": counts}
    (out / "results.json").write_text(json.dumps({"protocol": protocol, "runs": runs}, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
