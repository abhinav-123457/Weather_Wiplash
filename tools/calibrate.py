#!/usr/bin/env python3
"""
calibrate.py - fit CLIP_TEMPERATURE, DRY_THRESHOLD and DAMP_THRESHOLD to
labelled images.

This is CALIBRATION, NOT TRAINING. No model weights are touched. It scores
each image once with the production scorer, then searches for the three
constants that classify the set most accurately.

    calibrate/
        dry/    *.jpg
        damp/   *.jpg
        wet/    *.jpg

THREE CLASSES ONLY. There is deliberately no drying/ folder - drying is a
trend across frames and cannot be labelled from a single image. That is the
whole premise of the system, and a "drying" folder would be a noise class.

CROP EACH IMAGE TO MOSTLY TRACK SURFACE before saving it. No ROI is applied
here, so the image IS the region scored. This keeps calibration a measurement
of CLIP rather than of your cropping.

    .\\.venv\\Scripts\\python.exe tools\\calibrate.py
    .\\.venv\\Scripts\\python.exe tools\\calibrate.py --holdout 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.models.clip_scorer import ClipScorer  # noqa: E402

CLASSES = ["dry", "damp", "wet"]
TEMPERATURES = [0.002, 0.003, 0.005, 0.0075, 0.01, 0.015,
                0.02, 0.03, 0.05, 0.08, 0.12]


def camera_of(path: Path) -> str:
    """Infer camera type from the filename.

    Onboard sits metres from the surface and sees reflections and water
    texture directly; trackside averages the same surface over far fewer
    pixels at distance. The two can score differently for identical wetness,
    so thresholds fitted on one may misread the other.

    Put 'onboard' or 'trackside' anywhere in the filename to tag it.
    """
    n = path.name.lower()
    if "onboard" in n:
        return "onboard"
    if "trackside" in n or "aerial" in n:
        return "trackside"
    return "untagged"


def load_set(root: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for cls in CLASSES:
        folder = root / cls
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                items.append((p, cls))
    return items


def classify(score: float, t1: float, t2: float) -> str:
    if score < t1:
        return "dry"
    if score < t2:
        return "damp"
    return "wet"


def best_thresholds(scores: np.ndarray, labels: list[str]
                    ) -> tuple[float, float, float]:
    """Grid-search the two cut points that maximise accuracy.

    Ties are broken toward the WIDEST margin between the cuts and the nearest
    scores on either side. Two threshold pairs can score identically on a
    small set while one sits hard against a data point and the other sits
    mid-gap; the mid-gap pair generalises better.
    """
    lo, hi = float(scores.min()), float(scores.max())
    grid = np.linspace(max(lo - 2, 0), min(hi + 2, 100), 120)

    best = (0.0, -1.0, -1.0, -1.0)      # acc, margin, t1, t2
    for i, t1 in enumerate(grid):
        for t2 in grid[i + 1:]:
            pred = [classify(s, t1, t2) for s in scores]
            acc = float(np.mean([p == l for p, l in zip(pred, labels)]))
            margin = float(min(np.min(np.abs(scores - t1)),
                               np.min(np.abs(scores - t2))))
            if (acc, margin) > (best[0], best[1]):
                best = (acc, margin, float(t1), float(t2))

    return best[2], best[3], best[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("calibrate"))
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="fraction held out to report honest accuracy (e.g. 0.3)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = load_set(args.dir)
    if not items:
        sys.exit(f"No images found under {args.dir}/[dry|damp|wet]/")

    counts = {c: sum(1 for _, l in items if l == c) for c in CLASSES}
    print(f"{len(items)} images:  " +
          "  ".join(f"{c}={counts[c]}" for c in CLASSES))
    if min(counts.values()) < 5:
        print("  ! fewer than 5 in some class - thresholds will be fragile")
    print()

    print("loading CLIP...")
    scorer = ClipScorer()

    # Score once. Temperature is applied afterwards, so the sweep is free.
    print(f"scoring {len(items)} images...")
    sims, labels, paths = [], [], []
    for i, (path, label) in enumerate(items, 1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"  ! skipping {path.name}: {exc}")
            continue
        sims.append(scorer.similarities(img))
        labels.append(label)
        paths.append(path)
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")
    sims_arr = np.stack(sims)
    print()

    # ---- split ----
    idx = np.arange(len(labels))
    if args.holdout > 0:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(idx)
        n_test = max(1, int(len(idx) * args.holdout))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
    else:
        train_idx = test_idx = idx

    # ---- sweep temperature ----
    print(f"{'temp':>8}{'acc':>8}{'dry<':>8}{'damp<':>8}   spread per class")
    print("-" * 78)

    best = None
    for temp in TEMPERATURES:
        scores = np.array([
            scorer.wetness_from_similarities(s, temp)[0] for s in sims_arr])
        tr_scores = scores[train_idx]
        tr_labels = [labels[i] for i in train_idx]

        t1, t2, acc = best_thresholds(tr_scores, tr_labels)

        spread = []
        for c in CLASSES:
            vals = scores[[i for i in train_idx if labels[i] == c]]
            spread.append(f"{c} {vals.min():.0f}-{vals.max():.0f}"
                          if len(vals) else f"{c} -")

        print(f"{temp:>8.4f}{acc:>8.2f}{t1:>8.1f}{t2:>8.1f}   "
              + "  ".join(spread))

        if best is None or acc > best[0]:
            best = (acc, temp, t1, t2, scores)

    acc, temp, t1, t2, scores = best
    print()

    # ---- honest accuracy on unseen images ----
    if args.holdout > 0:
        pred = [classify(scores[i], t1, t2) for i in test_idx]
        true = [labels[i] for i in test_idx]
        hold_acc = float(np.mean([p == t for p, t in zip(pred, true)]))
        print(f"HOLDOUT accuracy: {hold_acc:.2f} on {len(test_idx)} unseen "
              f"images  (fitted accuracy {acc:.2f})")
        print("Quote the holdout number, never the fitted one - thresholds")
        print("chosen on a set will always look good on that same set.\n")

    # ---- confusion + misses ----
    print("confusion (rows = true, cols = predicted)")
    print(f"{'':>8}" + "".join(f"{c:>8}" for c in CLASSES))
    for tc in CLASSES:
        row = [sum(1 for i in idx
                   if labels[i] == tc and classify(scores[i], t1, t2) == pc)
               for pc in CLASSES]
        print(f"{tc:>8}" + "".join(f"{v:>8}" for v in row))

    misses = [(paths[i], labels[i], classify(scores[i], t1, t2), scores[i])
              for i in idx if classify(scores[i], t1, t2) != labels[i]]
    if misses:
        print(f"\n{len(misses)} misclassified:")
        for p, true, pred, s in sorted(misses, key=lambda m: m[3]):
            print(f"  {s:6.1f}  {true:>5} -> {pred:<5}  {p.name}")

    # ---- camera-type bias ----
    cams = [camera_of(p) for p in paths]
    present = sorted({c for c in cams if c != "untagged"})
    if len(present) >= 2:
        print("\nmedian score by camera type")
        print(f"{'':>10}" + "".join(f"{c:>12}" for c in present) + "     delta")
        worst = 0.0
        for cls in CLASSES:
            meds, row = [], f"{cls:>10}"
            for cam in present:
                vals = [scores[i] for i in idx
                        if labels[i] == cls and cams[i] == cam]
                if vals:
                    m = float(np.median(vals))
                    meds.append(m)
                    row += f"{m:>12.1f}"
                else:
                    row += f"{'-':>12}"
            if len(meds) >= 2:
                d = max(meds) - min(meds)
                worst = max(worst, d)
                row += f"{d:>10.1f}"
            print(row)

        if worst > 15:
            print(f"\n! CAMERA BIAS: up to {worst:.0f} points between camera "
                  f"types for the SAME class.")
            print("  Thresholds fitted across both will misread one of them.")
            print("  Either calibrate only on the type you will demo with,")
            print("  or keep separate thresholds per camera_type.")
        else:
            print(f"\n  camera types agree within {worst:.0f} points - "
                  f"one threshold set is fine")
    elif present:
        print(f"\n  all tagged images are '{present[0]}' - thresholds apply "
              f"to that camera type only")
    else:
        print("\n  no camera tags found. Put 'onboard' or 'trackside' in "
              "filenames to check for bias between them.")

    # ---- output ----
    print("\n" + "=" * 62)
    print("Paste into backend/app/config.py:\n")
    print(f"CLIP_TEMPERATURE = {temp}")
    print(f"DRY_THRESHOLD  = {t1:.1f}")
    print(f"DAMP_THRESHOLD = {t2:.1f}")
    print(f"DRYING_MIN_WETNESS = {t1:.1f}")
    print("=" * 62)

    gap = min(abs(t1 - scores[idx].min()), abs(t2 - scores[idx].max()))
    if len(items) < 20:
        print(f"\n! {len(items)} images is thin. 25-30 makes these stable.")
    print(f"! nearest score to a threshold: {gap:.1f} points"
          if gap < 5 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
