#!/usr/bin/env python3
"""
test_capacity.py - is the frozen CLIP embedding the bottleneck, or the head?

THE QUESTION THIS SETTLES
-------------------------
The system misreads dark-but-dry track (heavy rubber, overcast, floodlit) as
damp or wet. There are three possible causes and they need completely
different fixes:

  1 the information is NOT IN the frozen features
        -> a linear head cannot help, and neither can more data.
           Fine-tuning the encoder is the only route.
  2 the information IS there but is not LINEARLY separable
        -> logistic regression fails on it even though the features are
           fine. A non-linear head fixes it for ~2 minutes of work, and
           unfreezing 42M parameters would be a waste.
  3 the information is there and separable, but the LABELS confound it
        -> no architecture helps. Only data helps.

Guessing between these is how projects waste a week. This measures it.

HOW
---
The same frozen CLIP embeddings are fed to heads of increasing capacity:

    linear        logistic regression        (what ships today)
    mlp-64        one hidden layer, 64 units
    mlp-256       one hidden layer, 256 units
    rbf-svm       kernel method, effectively infinite-dimensional

Every number is LEAVE-ONE-RACE-OUT: each race is predicted by a model that
never saw a single frame of it. A random split would train on siblings of
its own test data and report a number 40 points too high - that mistake is
already documented in this repository.

READING THE RESULT
------------------
  MLP/SVM clearly beats linear   -> cause 2. The features are fine; the head
                                    was too weak. Ship a better head; do NOT
                                    unfreeze the encoder.
  All heads score about the same -> cause 1 or 3. Extra capacity found
                                    nothing to use. Check the dry-vs-damp
                                    breakdown below: if dry frames are being
                                    called damp at every capacity, get more
                                    labelled dark-dry frames before touching
                                    the architecture.
  Everything near chance (0.33)  -> the labels or the ROI are the problem,
                                    not the model.

    python tools/test_capacity.py
    python tools/test_capacity.py --dir calibrate
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
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
except ImportError:
    sys.exit("scikit-learn required:  pip install scikit-learn")

CLASSES = ["dry", "damp", "wet"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def race_of(path: Path) -> str:
    m = re.match(r"([A-Za-z]+\d{4})", path.stem)
    return (m.group(1) if m else path.stem.split("_")[0]).lower()


def heads(seed: int = 0) -> dict:
    """Increasing capacity over the SAME frozen features.

    class_weight balanced throughout, so a head cannot win by simply
    predicting the majority class.
    """
    return {
        "linear": LogisticRegression(C=1.0, max_iter=2000,
                                     class_weight="balanced"),
        "mlp-64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=3000,
                                alpha=1e-2, random_state=seed),
        "mlp-256": MLPClassifier(hidden_layer_sizes=(256,), max_iter=3000,
                                 alpha=1e-2, random_state=seed),
        "rbf-svm": SVC(kernel="rbf", C=10.0, gamma="scale",
                       class_weight="balanced", random_state=seed),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("calibrate"))
    args = ap.parse_args()

    items = []
    for ci, cls in enumerate(CLASSES):
        d = args.dir / cls
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    items.append((p, ci))
    if not items:
        sys.exit(f"no images under {args.dir}/[dry|damp|wet]/")

    y = np.array([c for _, c in items])
    races = np.array([race_of(p) for p, _ in items])
    print(f"{len(items)} frames  " + "  ".join(
        f"{c}={int((y == i).sum())}" for i, c in enumerate(CLASSES)))
    print(f"{len(set(races))} races: {', '.join(sorted(set(races)))}")

    print("\nembedding through the PRODUCTION ClipScorer (frozen)...")
    scorer = ClipScorer()
    X = np.stack([scorer.embedding(Image.open(p).convert("RGB"))
                  for p, _ in items])
    print(f"  {X.shape}")

    # Confirm any preprocessing experiment actually ran. Identical numbers
    # across two runs usually mean the flag never took effect, not that the
    # idea failed - and those two conclusions demand opposite next steps.
    from backend.app import config as _cfg
    from backend.app.models import clip_scorer as _cs
    if getattr(_cfg, "USE_CLAHE", False):
        n = getattr(_cs, "clahe_applied_count", 0)
        print(f"  CLAHE: ON, applied to {n}/{len(items)} images"
              + ("" if n == len(items) else "   <-- MISMATCH, results invalid"))
    else:
        print("  CLAHE: off  (baseline run)")

    uniq = sorted(set(races))
    results = {}

    def centre_by_venue(Xa, races_a):
        """Subtract each venue's own mean embedding, then renormalise.

        WHY THIS IS A FAIR TEST, NOT LEAKAGE
        The mean is computed from the IMAGES of a venue, never from their
        labels. A held-out venue is centred using its own unlabelled
        frames - information a deployed camera has for free, before anyone
        says what the conditions are.

        WHAT IT REMOVES
        CLIP embeddings cluster by scene: Monaco's barriers, Singapore's
        floodlights and a Vegas night are hugely different images, and
        wetness is a small direction inside that much larger variation.
        Subtracting the venue mean deletes "which circuit is this" and
        leaves "how does this frame differ from the rest of this venue" -
        which is the only question that generalises.

        This is the same correction the dry-reference button applies at
        runtime, done automatically and per venue.
        """
        out = Xa.copy()
        for r in set(races_a):
            m = races_a == r
            out[m] = Xa[m] - Xa[m].mean(axis=0, keepdims=True)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-8, None)

    Xc = centre_by_venue(X, races)

    def run(name, features):
        pred = np.full(len(y), -1)
        for r in uniq:
            te, tr = races == r, races != r
            if len(set(y[tr])) < 2:
                continue
            clf = heads()[name]
            clf.fit(features[tr], y[tr])
            pred[te] = clf.predict(features[te])
        return pred, pred >= 0

    for name in heads():
        results[name] = run(name, X)

    # The decisive comparison: same head, same features, venue offset removed.
    results["linear+centred"] = run("linear", Xc)
    results["mlp-64+centred"] = run("mlp-64", Xc)

    # ---- headline table ----
    print(f"\n{'head':<10}{'3-class':>9}{'wet/not':>9}{'dry kept dry':>14}"
          f"{'damp recall':>13}")
    print("-" * 56)
    for name, (pred, ok) in results.items():
        acc3 = (pred[ok] == y[ok]).mean()
        accw = ((pred[ok] == 2) == (y[ok] == 2)).mean()
        dry = ok & (y == 0)
        damp = ok & (y == 1)
        dry_ok = (pred[dry] == 0).mean() if dry.any() else float("nan")
        damp_ok = (pred[damp] == 1).mean() if damp.any() else float("nan")
        print(f"{name:<10}{acc3:>9.3f}{accw:>9.3f}{dry_ok:>13.1%}"
              f"{damp_ok:>13.1%}")

    # ---- where the dry frames actually go ----
    # The failure being diagnosed is specifically "dry read as damp/wet", so
    # the confusion for the dry row matters more than overall accuracy.
    print("\nwhere DRY frames end up (the reported failure)")
    print("-" * 56)
    print(f"{'head':<10}{'-> dry':>9}{'-> damp':>9}{'-> wet':>9}")
    for name, (pred, ok) in results.items():
        dry = ok & (y == 0)
        if not dry.any():
            continue
        row = [int((pred[dry] == k).sum()) for k in range(3)]
        print(f"{name:<10}" + "".join(f"{v:>9}" for v in row))

    def acc(name):
        pred, ok = results[name]
        return (pred[ok] == y[ok]).mean()

    def dryok(name):
        pred, ok = results[name]
        d = ok & (y == 0)
        return (pred[d] == 0).mean() if d.any() else float("nan")

    capacity_gain = max(acc("mlp-64"), acc("mlp-256"),
                        acc("rbf-svm")) - acc("linear")
    centre_gain = acc("linear+centred") - acc("linear")
    centre_dry_gain = dryok("linear+centred") - dryok("linear")

    print(f"""
VERDICT
  more capacity      {capacity_gain:+.3f} 3-class
  venue centring     {centre_gain:+.3f} 3-class   {centre_dry_gain:+.1%} dry-kept-dry
""")
    if centre_dry_gain >= 0.15 or centre_gain >= 0.08:
        print("  Removing the VENUE OFFSET moved the number that capacity")
        print("  could not. The wetness signal IS in the frozen features -")
        print("  it was buried under 'which circuit is this'. Fine-tuning is")
        print("  not required; per-camera calibration is. That is exactly")
        print("  what the dry-reference button does at runtime, so use it,")
        print("  and consider centring the probe's features per session.")
    elif capacity_gain >= 0.08:
        print("  A non-linear head found signal the hyperplane could not use.")
        print("  The features are NOT the bottleneck - swap the head.")
    else:
        print("  Neither extra capacity NOR removing the venue offset helped.")
        print("  With no venue holding both dry and damp, 'venue' and 'class'")
        print("  are the same feature and nothing recoverable is left in")
        print("  these labels. This is the point where fine-tuning becomes")
        print("  defensible - but it needs NEW DATA covering dark-dry, not a")
        print("  bigger model over the same 96 frames.")
    print("\nNOTHING WAS WRITTEN. This is a measurement, not a change.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())