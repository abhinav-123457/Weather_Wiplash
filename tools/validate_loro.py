#!/usr/bin/env python3
"""
validate_loro.py - leave-one-race-out validation.

WHY THIS EXISTS
---------------
The reported 83.3% came from a RANDOM 5-fold split over 96 images. Those
images are named by race:

    monaco2023_trackside_01..07     Saopaulo2024_trackside_06..15
    Singapore2025_onboard_01..06    US2025_trackside_17..22

Multiple frames per race, same session, same lighting, same camera. In CLIP
embedding space those are near-duplicates. A random split almost certainly
put siblings of a test frame into training - which inflates the score.

This holds out an ENTIRE RACE at a time, so nothing the model saw during
training came from the venue it is tested on.

WHY PER-RACE, NOT ONE AGGREGATE NUMBER
--------------------------------------
Race and class are confounded in this dataset: US2025 and Singapore2025 are
almost entirely dry, monaco2023 is damp and wet, Vegas2025 is wet. Holding
out a race therefore also removes most of a class, and some held-out folds
will not contain all three classes at all. A single pooled accuracy across
folds like that is not a meaningful number.

Per-race results stay interpretable even when a fold is unbalanced, and they
answer a more useful question: WHICH VENUES generalise and which do not.

THIS SCRIPT WRITES NOTHING. probe.npz and the existing 83.3% are untouched.

    python tools\\validate_loro.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.models.clip_scorer import ClipScorer  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, confusion_matrix
except ImportError:
    sys.exit("scikit-learn required:\n"
             "    .\\.venv\\Scripts\\pip.exe install scikit-learn")

CLASSES = ["dry", "damp", "wet"]

# Matched to the original run so any difference is attributable to the
# VALIDATION change, not to a different hyperparameter. Note that C=10 was
# itself chosen under the leaky CV, so it is mildly optimistic here - a fully
# clean protocol would re-select C inside each fold.
C = 10.0


def race_of(path: Path) -> str:
    """Everything before the first underscore, lowercased.

    monaco2023_trackside_01.png -> monaco2023
    Singapore2017_onboard_09.png -> singapore2017

    The year is part of the key on purpose: Singapore 2017 and Singapore 2025
    are different sessions, different cars, different broadcast era.
    """
    stem = path.stem
    m = re.match(r"([A-Za-z]+\d{4})", stem)
    if m:
        return m.group(1).lower()
    return stem.split("_")[0].lower()


def main() -> int:
    root = Path("calibrate")
    items: list[tuple[Path, int]] = []
    for ci, cls in enumerate(CLASSES):
        folder = root / cls
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    items.append((p, ci))
    if not items:
        sys.exit("no images under calibrate/[dry|damp|wet]/")

    races = [race_of(p) for p, _ in items]
    uniq = sorted(set(races))

    print("=" * 74)
    print("LEAVE-ONE-RACE-OUT VALIDATION")
    print("=" * 74)
    print(f"\n{len(items)} images across {len(uniq)} races\n")

    # ---- show the confounding explicitly, before any number appears ----
    print(f"{'race':<18}{'n':>4}   class mix")
    print("-" * 56)
    for r in uniq:
        idx = [i for i, rr in enumerate(races) if rr == r]
        c = Counter(CLASSES[items[i][1]] for i in idx)
        mix = "  ".join(f"{k}={c[k]}" for k in CLASSES if c[k])
        print(f"{r:<18}{len(idx):>4}   {mix}")

    print("\nloading CLIP (frozen)...")
    scorer = ClipScorer()

    print(f"extracting embeddings from {len(items)} images...")
    X, y, paths = [], [], []
    for i, (path, label) in enumerate(items, 1):
        try:
            X.append(scorer.embedding(Image.open(path).convert("RGB")))
            y.append(label)
            paths.append(path)
        except Exception as exc:
            print(f"  ! skip {path.name}: {exc}")
        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")
    X = np.stack(X)
    y = np.array(y)
    races = np.array([race_of(p) for p in paths])

    # ---- baseline: the original random split, same code path ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rand_pred = cross_val_predict(
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        X, y, cv=cv)
    rand_acc = accuracy_score(y, rand_pred)

    # ---- leave one race out ----
    print(f"\n{'held-out race':<18}{'n':>4}{'acc':>8}   classes present   notes")
    print("-" * 74)

    pooled_true, pooled_pred = [], []
    rows = []
    for r in uniq:
        te = races == r
        tr = ~te
        train_classes = set(y[tr])
        test_classes = sorted(set(y[te]))

        if len(train_classes) < 2:
            print(f"{r:<18}{te.sum():>4}{'skip':>8}   "
                  f"training set has < 2 classes")
            continue

        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        acc = accuracy_score(y[te], pred)

        pooled_true.extend(y[te])
        pooled_pred.extend(pred)

        present = ",".join(CLASSES[c] for c in test_classes)
        note = ""
        if len(test_classes) < 3:
            note = f"only {len(test_classes)}/3 classes - accuracy not comparable"
        missing_in_train = set(range(3)) - train_classes
        if missing_in_train:
            note = ("training set MISSING "
                    + ",".join(CLASSES[c] for c in missing_in_train))

        print(f"{r:<18}{te.sum():>4}{acc:>8.3f}   {present:<16}  {note}")
        rows.append((r, te.sum(), acc, present))

    pooled_true = np.array(pooled_true)
    pooled_pred = np.array(pooled_pred)
    loro_acc = accuracy_score(pooled_true, pooled_pred)

    # ---- comparison ----
    print("\n" + "=" * 74)
    print(f"  random 5-fold (as originally reported) : {rand_acc:.3f}")
    print(f"  leave-one-race-out (pooled)            : {loro_acc:.3f}")
    print(f"  difference                             : {loro_acc - rand_acc:+.3f}")
    print("=" * 74)

    if rows:
        accs = [a for _, _, a, _ in rows]
        print(f"\nper-race spread: {min(accs):.2f} to {max(accs):.2f}  "
              f"(median {sorted(accs)[len(accs)//2]:.2f})")
        worst = sorted(rows, key=lambda t: t[2])[:3]
        print("weakest venues: " + ", ".join(f"{r} {a:.2f}" for r, _, a, _ in worst))

    print("\nconfusion, pooled across held-out races")
    cm = confusion_matrix(pooled_true, pooled_pred, labels=range(3))
    print(f"{'':>8}" + "".join(f"{c:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>8}" + "".join(f"{v:>8}" for v in cm[i]))

    y2 = (pooled_true == 2).astype(int)
    p2 = (pooled_pred == 2).astype(int)
    print(f"\nwet vs not-wet (leave-one-race-out): "
          f"{accuracy_score(y2, p2):.3f}")

    # ---- misses, grouped by race ----
    wrong = np.where(pooled_pred != pooled_true)[0]
    if len(wrong):
        print(f"\n{len(wrong)} misclassified across all held-out races")

    print("""
HOW TO READ THIS
  The pooled leave-one-race-out number is a FLOOR, not a like-for-like
  replacement for the random-split figure. With ~6 races and class/venue
  confounding, a fold can lose most of a class, so the estimate is noisy -
  plausibly +/- 10 points on this much data.

  The per-race column is the more useful output: it says which venues the
  model generalises to and which it does not. That is a property of the
  DATASET as much as the model, and it points at what to collect next.

NOTHING WAS WRITTEN. probe.npz is unchanged.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
