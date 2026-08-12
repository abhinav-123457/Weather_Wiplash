#!/usr/bin/env python3
"""
inspect_bands.py - do spatial sub-bands carry a usable signal?

WHY BANDS AT ALL
----------------
Leave-one-race-out showed the absolute wetness score is not venue-invariant:
the per-venue offset (16-28 points) is as large as the between-class gap
(16.3 points at Monaco). No global threshold survives that.

Bands sidestep it. Every band in one frame shares the same camera, lighting,
venue and exposure, so band-to-band DIFFERENCES are venue-normalised by
construction. This is the one place the dataset's central weakness does not
apply.

WHAT IS MEASURED
  full score      the existing whole-ROI estimate, unchanged
  band scores     N horizontal bands, each scored independently
  median          a robust alternative to the full-ROI score
  range           |max - min| across bands
  monotonicity    fraction of adjacent steps agreeing in direction

Monotonic ordering, not variance. 72 > 65 > 49 > 31 is a gradient;
72, 31, 65, 49 has identical variance and means nothing.

A CROP DETAIL THAT MATTERS
CLIP's processor resizes the shortest side to 224 then centre-crops. A thin
full-width strip would therefore lose most of its width before the model
ever saw it. Each band is resized to square first so the whole strip reaches
CLIP. The full ROI is scored BOTH ways so any effect of that choice is
visible rather than assumed.

SCORING
Leave-one-race-out throughout - each image is scored by a model that never
saw its venue.

THIS SCRIPT WRITES NOTHING. It is a measurement, not a feature.

    python tools\\inspect_bands.py
    python tools\\inspect_bands.py --bands 5 --top 15
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.models.clip_scorer import ClipScorer  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    sys.exit("scikit-learn required")

CLASSES = ["dry", "damp", "wet"]
VALUES = np.array([0.0, 50.0, 100.0])
C = 10.0
SQUARE = 224


def race_of(path: Path) -> str:
    m = re.match(r"([A-Za-z]+\d{4})", path.stem)
    return (m.group(1) if m else path.stem.split("_")[0]).lower()


def bands_of(img: Image.Image, n: int) -> list[Image.Image]:
    """Split top-to-bottom into n horizontal bands, each resized to square.

    Deliberately called BANDS, not "near/far" or "racing line": the crop
    geometry varies between shots, so band index carries no guaranteed
    physical meaning. Naming it a dry line before verifying camera geometry
    would be assuming the conclusion.
    """
    w, h = img.size
    step = h / n
    out = []
    for i in range(n):
        top, bot = int(i * step), int((i + 1) * step)
        out.append(img.crop((0, top, w, bot)).resize((SQUARE, SQUARE)))
    return out


def monotonicity(vals: list[float]) -> float:
    """Fraction of adjacent steps agreeing with the overall direction."""
    if len(vals) < 2:
        return 0.0
    diffs = np.diff(vals)
    if np.all(diffs == 0):
        return 0.0
    direction = np.sign(vals[-1] - vals[0])
    if direction == 0:
        return 0.0
    return float(np.mean(np.sign(diffs) == direction))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--top", type=int, default=12,
                    help="how many strongest gradients to list for inspection")
    args = ap.parse_args()

    root = Path("calibrate")
    items = []
    for ci, cls in enumerate(CLASSES):
        d = root / cls
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    items.append((p, ci))
    if not items:
        sys.exit("no images under calibrate/[dry|damp|wet]/")

    print("=" * 78)
    print(f"SPATIAL BAND INSPECTION  ({args.bands} bands, LORO scoring)")
    print("=" * 78)

    print("\nloading CLIP (frozen)...")
    scorer = ClipScorer()
    print(f"embedding {len(items)} images x ({args.bands} bands + 2 full)...")

    emb_full, emb_full_sq, emb_bands, y, paths = [], [], [], [], []
    for i, (p, lab) in enumerate(items, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as exc:
            print(f"  ! skip {p.name}: {exc}")
            continue
        emb_full.append(scorer.embedding(img))                       # production path
        emb_full_sq.append(scorer.embedding(img.resize((SQUARE, SQUARE))))
        emb_bands.append(np.stack([scorer.embedding(b)
                                   for b in bands_of(img, args.bands)]))
        y.append(lab); paths.append(p)
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    emb_full = np.stack(emb_full)
    emb_full_sq = np.stack(emb_full_sq)
    emb_bands = np.stack(emb_bands)          # (N, bands, 512)
    y = np.array(y)
    races = np.array([race_of(p) for p in paths])
    uniq = sorted(set(races))

    def score(clf, E):
        pr = clf.predict_proba(E)
        full = np.zeros((pr.shape[0], 3))
        for c, i in {c: i for i, c in enumerate(clf.classes_)}.items():
            full[:, c] = pr[:, i]
        return full @ VALUES, full.max(axis=1)

    n = len(y)
    s_full = np.full(n, np.nan)
    s_full_sq = np.full(n, np.nan)
    s_bands = np.full((n, args.bands), np.nan)
    conf_bands = np.full((n, args.bands), np.nan)

    for r in uniq:
        te, tr = races == r, races != r
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(emb_full[tr], y[tr])
        s_full[te], _ = score(clf, emb_full[te])
        s_full_sq[te], _ = score(clf, emb_full_sq[te])
        for b in range(args.bands):
            s_bands[te, b], conf_bands[te, b] = score(clf, emb_bands[te, b])

    med = np.nanmedian(s_bands, axis=1)
    rng = np.nanmax(s_bands, axis=1) - np.nanmin(s_bands, axis=1)
    mono = np.array([monotonicity(list(s_bands[i])) for i in range(n)])

    # ---- does the square resize change the full-ROI score? ----
    d = np.abs(s_full - s_full_sq)
    print(f"\nfull ROI: production crop vs square resize")
    print(f"  mean |difference| {np.nanmean(d):.1f} points, "
          f"max {np.nanmax(d):.1f}")
    print("  (large values here would mean band scores are not comparable")
    print("   to the production score, only to each other)")

    # ---- per race / class ----
    print(f"\n{'race':<15}{'class':<6}{'n':>3}{'full':>7}{'median':>8}"
          f"{'range':>7}{'mono':>6}{'conf':>6}")
    print("-" * 78)
    for r in uniq:
        for ci, cls in enumerate(CLASSES):
            m = (races == r) & (y == ci) & ~np.isnan(s_full)
            if not m.any():
                continue
            print(f"{r:<15}{cls:<6}{m.sum():>3}{np.nanmean(s_full[m]):>7.1f}"
                  f"{np.nanmean(med[m]):>8.1f}{np.nanmean(rng[m]):>7.1f}"
                  f"{np.nanmean(mono[m]):>6.2f}"
                  f"{np.nanmean(conf_bands[m]):>6.2f}")
        print()

    # ---- is the band median a better estimator than the full score? ----
    ok = ~np.isnan(s_full)
    print("band median vs full-ROI score, as a class separator")
    print("-" * 78)
    for name, vals in (("full ROI", s_full), ("band median", med)):
        line = f"  {name:<13}"
        for ci, cls in enumerate(CLASSES):
            m = ok & (y == ci)
            line += f"{cls} {np.nanmean(vals[m]):5.1f}   "
        # Monaco is the only venue where a within-venue gap is measurable.
        mo = races == "monaco2023"
        a = vals[mo & (y == 1)]; b = vals[mo & (y == 2)]
        if len(a) and len(b):
            pooled = np.sqrt((np.nanstd(a) ** 2 + np.nanstd(b) ** 2) / 2) or 1e-9
            line += f"| monaco damp->wet {abs(np.nanmean(b) - np.nanmean(a)) / pooled:.2f} SD"
        print(line)

    # ---- monotonicity by class ----
    print("\nmonotonicity by class  (1.00 = perfectly ordered gradient)")
    print("-" * 78)
    for ci, cls in enumerate(CLASSES):
        m = ok & (y == ci)
        strong = np.mean((mono[m] >= 0.99) & (rng[m] >= 15))
        print(f"  {cls:<6} mean {np.nanmean(mono[m]):.2f}   "
              f"range {np.nanmean(rng[m]):5.1f}   "
              f"strong gradient in {strong * 100:4.0f}% of frames")

    # ---- the images to actually look at ----
    print(f"\nstrongest gradients - OPEN THESE AND CHECK THEM BY EYE")
    print("A monotonic sequence can come from a real wet/dry transition, or")
    print("from perspective, lighting falloff, rubber, kerbs or a shadow.")
    print("The mathematics cannot tell them apart; your eyes can.")
    print("-" * 78)
    order = np.argsort(-np.nan_to_num(rng))[:args.top]
    print(f"{'file':<34}{'class':<6}{'range':>7}{'mono':>6}   bands")
    for i in order:
        bands = " ".join(f"{v:4.0f}" for v in s_bands[i])
        print(f"{paths[i].name[:33]:<34}{CLASSES[y[i]]:<6}"
              f"{rng[i]:>7.1f}{mono[i]:>6.2f}   {bands}")

    print("""
HOW TO DECIDE FROM THIS
  Band median beats full ROI on the Monaco separation  -> use it as the score
  Monotonicity near 1.0 with large range, only on some frames -> a real
    spatial signal exists and a conservative gradient feature is justified
  Monotonicity high on nearly ALL frames, including dry -> it is measuring
    perspective or lighting, not water. Do NOT build a dry-line detector.
  Range small everywhere -> no spatial structure to detect. Stop here.

NOTHING WAS WRITTEN.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
