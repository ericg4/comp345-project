#!/usr/bin/env python3
"""Train the CNN with synthetic ink planted as a shortcut, then test with it removed.

    uv run python scripts/download_datasets.py --ham10000-segmentations --extract
    uv run python scripts/ink_shortcut.py

Uses the ink designs from synthetic_artifacts.py (the SmolVLM insertion study).
Every eligible image gets one fixed ink stamp (design, rotation, position) at least
15 native pixels from every lesion-mask pixel, and inside the region the CNN's
center crop keeps. (The SmolVLM study protected the lesion's whole bounding box;
here that left too few large lesions, mostly melanomas, with room for ink.) Images
whose lesion mask has foreground at a corner, or with no safe position, are
excluded from every split and level.

At each level (melanoma share, nevus share), exactly that share of each class in
train and val carries its stamp. One fixed random draw per image decides who gets
ink, so a stronger level inks a superset of the images of a weaker one. Each level
trains the CNN with several seeds, choosing epoch and threshold on the equally
biased val split, then scores the test split with ink as in training (biased), on
no image (clean), on every image (all), and with the class shares swapped
(flipped). The existing data/cnn/<size>.pt model, trained without ink, is scored on
clean and all as a baseline. Writes results/cnn/<size>/ink_shortcut/.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import PIL
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from cnn import CACHE_MAX_SIZE, CHANNELS, SmallCNN, depth, make_loader, pick_device, predict, train, transforms_for
from common import RESULTS_DIR, ROOT, load_split, metrics
from synthetic_artifacts import composite, paired_overlays, seed_for

MASK_DIR = (ROOT / "data" / "ham10000_segmentations" / "HAM10000_segmentations_lesion_tschandl"
            / "HAM10000_segmentations_lesion_tschandl")
SEED = 42
# Native pixels kept free of ink around every lesion pixel (the SmolVLM study's margin).
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
        (tn, fp), (fn, tp) = score["confusion_matrix"]
        score.update(sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                     mel_ink_rate=float(ink[labels == 1].mean()), nv_ink_rate=float(ink[labels == 0].mean()))
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
    start = time.time()
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

    preview = []
    for dx in ("nv", "mel"):
        for row in test[test["dx"] == dx].head(2).itertuples():
            native = Image.open(row.path).convert("RGB")
            native = Image.fromarray(composite(np.array(native), STAMPS[row.design, row.rotation], row.x, row.y))
            small = transforms_for(args.size)[2].transforms[0](inked[row.Index])  # The center crop the CNN sees.
            preview.append((native.resize((400, 300)), small.resize((300, 300), Image.Resampling.NEAREST)))
    sheet = Image.new("RGB", (720, 310 * len(preview)), "white")
    for k, (native, small) in enumerate(preview):
        sheet.paste(native, (0, 310 * k))
        sheet.paste(small, (410, 310 * k))
    sheet.save(out / "preview.png")

    summary, runs, predictions = [], [], []

    def record(model_name, seed, level, scores, effects, outputs):
        for name, score in scores.items():
            summary.append({"model": model_name, "seed": seed, "condition": name, "n": len(test),
                            **{k: score[k] for k in ("mel_ink_rate", "nv_ink_rate", "macro_f1", "auroc",
                                                     "sensitivity", "specificity", "flip_rate_vs_clean")},
                            "mel_f1": score["f1"]["mel"]})
            probs, margins = outputs[name]
            predictions.append(pd.DataFrame({"model": model_name, "seed": seed, "condition": name,
                                             "image_id": test["image_id"].to_numpy(), "label": test["label"].to_numpy(),
                                             "prob_mel": probs[:, 1], "margin": margins}))
        runs.append({"model": model_name, "seed": seed, "level": level, "test": scores, "effects": effects})

    baseline = SmallCNN(CHANNELS[:depth(args.size)]).to(pick_device())
    baseline.load_state_dict(torch.load(ROOT / "data" / "cnn" / f"{args.size}.pt", map_location=pick_device(),
                                        weights_only=True))
    threshold = json.loads((RESULTS_DIR / "cnn" / str(args.size) / "trained.json").read_text())["chosen"]["threshold"]
    conditions = {"clean": np.zeros(len(test), bool), "all": np.ones(len(test), bool)}
    scores, effects, outputs = evaluate(baseline, threshold, test, clean, inked, conditions, args.size)
    record("baseline", None, None, scores, effects, outputs)
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
            checkpoint = ROOT / "data" / "cnn" / "ink_shortcut" / f"{args.size}_{level.replace('/', '-')}_seed{seed}.pt"
            model, best, history = train(images, labels, parts, args.size, checkpoint, epochs=args.epochs, seed=seed,
                                         verbose=False)
            scores, effects, outputs = evaluate(model, best["threshold"], test, clean, inked, conditions, args.size)
            record(level, seed, level, scores, effects, outputs)
            runs[-1].update(chosen=best, history=history)
            print(f"{level} seed {seed}: val F1 {best['val_macro_f1']:.3f}; test macro F1 "
                  + ", ".join(f"{name} {s['macro_f1']:.3f}" for name, s in scores.items())
                  + f"; ink margin shift mel {effects['mel']['all_minus_clean_margin']:+.2f}, "
                    f"nv {effects['nv']['all_minus_clean_margin']:+.2f}")

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.concat(predictions).to_csv(out / "predictions.csv.gz", index=False)
    counts = {f"{split}/{dx}/{reason or 'eligible'}": int(n) for (split, dx), row in exclusions.iterrows()
              for reason, n in row.items() if n}
    protocol = {"seed": SEED, "margin_native_pixels": MARGIN, "crop_fraction": CROP_FRACTION, "levels": LEVELS,
                "bootstrap": BOOTSTRAP, "size": args.size, "seeds": args.seeds, "epochs": args.epochs,
                "n_eligible": len(eligible), "counts": counts}
    (out / "results.json").write_text(json.dumps({"protocol": protocol, "runs": runs}, indent=2) + "\n")
    (out / "run_info.json").write_text(json.dumps({
        "command": "scripts/ink_shortcut.py " + " ".join(f"--{k} {v}" for k, v in vars(args).items()),
        "device": pick_device(), "elapsed_seconds": time.time() - start,
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "pillow": PIL.__version__}}, indent=2) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
