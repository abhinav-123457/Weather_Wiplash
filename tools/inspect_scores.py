#!/usr/bin/env python3
"""
inspect_scores.py - does the CONTINUOUS wetness signal survive where the
3-class labels do not?

WHY
---
Leave-one-race-out gave 36.5% on three classes but 85.4% on wet-vs-not-wet.
The confusion matrix showed dry collapsing into damp (1 of 31 correct) when
the venue changed.

That is a statement about ARGMAX. It says nothing about whether the
underlying score is still ordered. A model can be useless at naming the
class while still producing wet > damp > dry within a venue - and if that
ordering holds, the temporal layer can work off it, because a trend only
needs the score to MOVE correctly, not to sit at the right absolute level.

This measures that directly, and it also quantifies how much of the failure
is a fixed per-venue OFFSET rather than scrambled ordering. That distinction
matters: an offset is something a one-time per-camera calibration could
absorb, and a trackside camera lives at one circuit.

WHAT IT CANNOT ANSWER
---------------------
No venue in this dataset contains both dry AND damp. Monaco is the only
multi-class venue and it holds damp + wet. So the within-venue ordering test
can only check damp < wet - which is the distinction that already works.
The dry/damp ordering is simply not testable with this data, and a good
Monaco result must not be read as evidence that it is fine.

SCORING PROTOCOL
----------------
Every image is scored by a model trained WITHOUT its own race. Using the
full probe would flatter every venue it had already seen.

THIS SCRIPT WRITES NOTHING.

    python tools\\inspect_scores.py
"""

from __future__ import annotations

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


def race_of(path: Path) -> str:
    m = re.match(r"([A-Za-z]+\d{4})", path.stem)
    return (m.group(1) if m else path.stem.split("_")[0]).lower()


def main() -> int:
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

    print("=" * 74)
    print("CONTINUOUS SCORE INSPECTION  (leave-one-race-out scoring)")
    print("=" * 74)

    print("\nloading CLIP (frozen)...")
    scorer = ClipScorer()
    print(f"extracting embeddings from {len(items)} images...")

    X, y, paths = [], [], []
    for i, (p, lab) in enumerate(items, 1):
        try:
            X.append(scorer.embedding(Image.open(p).convert("RGB")))
            y.append(lab); paths.append(p)
        except Exception as exc:
            print(f"  ! skip {p.name}: {exc}")
        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    X = np.stack(X); y = np.array(y)
    races = np.array([race_of(p) for p in paths])
    uniq = sorted(set(races))

    # ---- score every image with a model that never saw its race ----
    wetness = np.zeros(len(y))
    confidence = np.zeros(len(y))
    for r in uniq:
        te, tr = races == r, races != r
        if len(set(y[tr])) < 2:
            wetness[te] = np.nan
            continue
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        # Map the model's own class order onto ours before weighting.
        cols = {c: i for i, c in enumerate(clf.classes_)}
        full = np.zeros((proba.shape[0], 3))
        for c, i in cols.items():
            full[:, c] = proba[:, i]
        wetness[te] = full @ VALUES
        confidence[te] = full.max(axis=1)

    # ---- per race, per class ----
    print(f"\n{'race':<16}{'class':<7}{'n':>4}{'mean':>8}{'median':>8}"
          f"{'std':>7}{'min':>7}{'max':>7}{'conf':>7}")
    print("-" * 74)
    table = {}
    for r in uniq:
        for ci, cls in enumerate(CLASSES):
            m = (races == r) & (y == ci) & ~np.isnan(wetness)
            if not m.any():
                continue
            w = wetness[m]
            table[(r, cls)] = float(np.median(w))
            print(f"{r:<16}{cls:<7}{m.sum():>4}{w.mean():>8.1f}"
                  f"{np.median(w):>8.1f}{w.std():>7.1f}{w.min():>7.1f}"
                  f"{w.max():>7.1f}{confidence[m].mean():>7.2f}")
        print()

    # ---- the venue-offset matrix ----
    print("median wetness score by race x class  ('-' = class absent)")
    print(f"{'race':<16}" + "".join(f"{c:>9}" for c in CLASSES))
    print("-" * 44)
    for r in uniq:
        row = f"{r:<16}"
        for c in CLASSES:
            v = table.get((r, c))
            row += f"{v:>9.1f}" if v is not None else f"{'-':>9}"
        print(row)

    # ---- how far apart are venues for the SAME class? ----
    print("\nsame class, different venue - how big is the offset?")
    print("(a large spread here means the failure is largely a fixed per-venue")
    print(" shift, which a one-time camera calibration could absorb)")
    print("-" * 74)
    for c in CLASSES:
        vals = [(r, table[(r, c)]) for r in uniq if (r, c) in table]
        if len(vals) < 2:
            print(f"  {c:<6} only {len(vals)} venue - offset not measurable")
            continue
        lo = min(vals, key=lambda t: t[1]); hi = max(vals, key=lambda t: t[1])
        print(f"  {c:<6} spread {hi[1] - lo[1]:5.1f} points   "
              f"{lo[0]} {lo[1]:.1f}  ->  {hi[0]} {hi[1]:.1f}")

    # ---- within-venue ordering, where testable ----
    print("\nwithin-venue ordering (the only honest test of the signal)")
    print("-" * 74)
    testable = 0
    for r in uniq:
        present = [c for c in CLASSES if (r, c) in table]
        if len(present) < 2:
            continue
        testable += 1
        meds = [(c, table[(r, c)]) for c in present]
        order = " < ".join(f"{c} {v:.1f}" for c, v in meds)
        expected = [c for c in CLASSES if c in present]
        got = [c for c, _ in sorted(meds, key=lambda t: t[1])]
        ok = got == expected
        print(f"  {r}: {order}")
        print(f"    expected {' < '.join(expected)}  ->  "
              f"{'ORDER HOLDS' if ok else 'ORDER BROKEN'}")

        # Separation relative to spread tells us whether the ordering is
        # usable or merely technically correct.
        if len(present) == 2:
            a = wetness[(races == r) & (y == CLASSES.index(present[0]))]
            b = wetness[(races == r) & (y == CLASSES.index(present[1]))]
            pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2) or 1e-9
            print(f"    separation {abs(b.mean() - a.mean()) / pooled:.2f} "
                  f"pooled SDs  (>1.0 is a usable gap)")

    if testable == 0:
        print("  none - no venue in this dataset contains two classes")

    print("""
READING THIS
  If ordering holds within a venue but the absolute level shifts between
  venues, the signal is intact and the problem is CALIBRATION, not vision.
  That is good news for a per-venue deployment and for the temporal layer,
  which reads movement rather than level.

  If ordering breaks within a venue, the score is not tracking wetness and
  no amount of calibration or smoothing rescues it.

  Remember what is NOT tested: no venue here has both dry and damp, so the
  distinction that actually failed cannot be checked. A good Monaco result
  says damp < wet holds. It says nothing about dry < damp.

NOTHING WAS WRITTEN.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
