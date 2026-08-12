#!/usr/bin/env python3
"""
train_probe.py - fit a linear probe on frozen CLIP embeddings.

WHAT THIS IS AND IS NOT
-----------------------
NOT fine-tuning. CLIP's 88M weights are frozen and never touched. This fits
a logistic regression - roughly 1,500 parameters - on the 512-d embeddings
CLIP already produces. It is the standard way to adapt CLIP to a specific
task with ~100 labelled examples, and it runs on CPU in seconds.

WHY
---
Zero-shot prompting scored 54% holdout on 96 frames. Two prompt designs were
tried; the more elaborate one generalised WORSE. Dry spanned 7-63 and damp
17-80 - near-total overlap - with a 19-point swing between camera types for
the same class. That is viewpoint driving the score more than water does,
and no amount of rewording fixes it.

This script also answers a question prompting cannot: IS THE INFORMATION
EVEN THERE? If a probe trained directly on the labels still cannot separate
dry from damp, then the distinction is not present in the CLIP features, and
the honest response is to report wet/not-wet confidently and let the trend
disambiguate the rest.

    python tools\\train_probe.py
    python tools\\train_probe.py --dir calibrate --out backend/app/probe.npz
"""

from __future__ import annotations

import argparse
import sys
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
VALUES = np.array([0.0, 50.0, 100.0])


def camera_of(path: Path) -> str:
    n = path.name.lower()
    if "onboard" in n:
        return "onboard"
    if "trackside" in n or "aerial" in n:
        return "trackside"
    return "untagged"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("calibrate"))
    ap.add_argument("--out", type=Path,
                    default=Path("backend/app/probe.npz"))
    ap.add_argument("--C", type=float, default=0.0,
                    help="inverse regularisation; 0 = sweep and pick best")
    args = ap.parse_args()

    items: list[tuple[Path, str]] = []
    for ci, cls in enumerate(CLASSES):
        folder = args.dir / cls
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    items.append((p, cls))
    if not items:
        sys.exit(f"no images under {args.dir}/[dry|damp|wet]/")

    print(f"{len(items)} images: " + "  ".join(
        f"{c}={sum(1 for _, l in items if l == c)}" for c in CLASSES))

    print("\nloading CLIP (frozen - weights are never modified)...")
    scorer = ClipScorer()

    print(f"extracting embeddings from {len(items)} images...")
    X, y, paths = [], [], []
    for i, (path, label) in enumerate(items, 1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"  ! skip {path.name}: {exc}")
            continue
        X.append(scorer.embedding(img))
        y.append(CLASSES.index(label))
        paths.append(path)
        if i % 20 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    X = np.stack(X)
    y = np.array(y)
    print(f"\nembeddings: {X.shape}")

    # ---- cross-validated regularisation sweep ----
    # 512 features against ~96 samples overfits trivially without strong
    # regularisation, so C is chosen by cross-validation rather than guessed.
    # Every accuracy printed is from predictions on held-out folds.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    candidates = [args.C] if args.C > 0 else [0.003, 0.01, 0.03, 0.1,
                                              0.3, 1.0, 3.0, 10.0]

    print(f"\n{'C':>8}{'cv_acc':>10}   (5-fold, held-out predictions)")
    print("-" * 40)
    best = None
    for C in candidates:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        pred = cross_val_predict(clf, X, y, cv=cv)
        acc = accuracy_score(y, pred)
        print(f"{C:>8.3f}{acc:>10.3f}")
        if best is None or acc > best[0]:
            best = (acc, C, pred)

    acc, C, pred = best
    print(f"\nbest: C={C}  cross-validated accuracy {acc:.3f}")
    print("This is an honest number - every prediction came from a model")
    print("that had not seen that image.\n")

    # ---- confusion ----
    cm = confusion_matrix(y, pred)
    print("confusion (rows = true, cols = predicted)")
    print(f"{'':>8}" + "".join(f"{c:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>8}" + "".join(f"{v:>8}" for v in cm[i]))

    # ---- is wet/not-wet easier? ----
    # If dry-vs-damp is genuinely not in the pixels, a two-class split
    # should still be strong, and that is a system worth shipping.
    y2 = (y == 2).astype(int)
    p2 = (pred == 2).astype(int)
    print(f"\nwet vs not-wet accuracy: {accuracy_score(y2, p2):.3f}")

    # ---- misclassifications ----
    wrong = np.where(pred != y)[0]
    if len(wrong):
        print(f"\n{len(wrong)} misclassified:")
        for i in wrong[:25]:
            print(f"  {CLASSES[y[i]]:>5} -> {CLASSES[pred[i]]:<5} "
                  f"[{camera_of(paths[i]):>9}]  {paths[i].name}")

    # ---- fit final model on everything and save ----
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, coef=clf.coef_, intercept=clf.intercept_,
             classes=np.array(CLASSES), values=VALUES, C=C, cv_accuracy=acc)

    print(f"\nsaved probe to {args.out}  ({clf.coef_.size} parameters)")
    print("Set USE_PROBE = True in backend/app/config.py to use it.")

    if acc < 0.70:
        print("\n! Under 70% even with a probe trained on the labels means")
        print("  the dry/damp distinction is largely absent from the")
        print("  features. Report wet/not-wet with confidence and let the")
        print("  TREND separate drying from wetting - which is what the")
        print("  temporal layer exists to do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
