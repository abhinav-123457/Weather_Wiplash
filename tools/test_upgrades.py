#!/usr/bin/env python3
"""
test_upgrades.py - re-test every proposed upgrade on the CURRENT dataset.

WHY THIS EXISTS
---------------
Most of the ideas people suggest for this project were already measured -
but they were measured on the ORIGINAL 96 frames, where five of six venues
held a single class. That dataset could not answer the question, so any
conclusion drawn from it is suspect.

The proof that this matters is already in hand. Venue centring was
"degenerate, ignore it" on 96 frames and is the BEST head on 199. Same
code, opposite verdict, because the data changed. So every earlier
rejection deserves re-testing rather than re-quoting.

This runs them all on the current calibrate/ set, under leave-one-race-out,
in one pass:

    clip                 the shipped baseline
    clip + centred       subtract each venue's own mean embedding
    clip + clahe         equalise lightness before embedding
    clip + augment       brightness/contrast/hue jitter, TRAIN FOLDS ONLY
    clip + aug + centred  both
    clip + cv            concatenate the classical CV signals
    dinov2               a different frozen backbone entirely

FAIRNESS RULES BAKED IN
  - every variant is scored leave-one-race-out on the same frames
  - augmentation is applied to TRAINING folds only; augmenting the test set
    would be measuring the augmentation, not the model
  - centring uses each venue's own IMAGES, never its labels
  - a variant that collapses to one class is flagged, not celebrated

WRITES NOTHING. It is a measurement.

    python tools/test_upgrades.py
    python tools/test_upgrades.py --skip dinov2       # if the download is slow
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    sys.exit("scikit-learn required")

CLASSES = ["dry", "damp", "wet"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def race_of(path: Path) -> str:
    m = re.match(r"([A-Za-z]+\d{4})", path.stem)
    return (m.group(1) if m else path.stem.split("_")[0]).lower()


def augment(img: Image.Image, rng) -> Image.Image:
    """Lighting jitter only - never anything that changes the WETNESS.

    Brightness, contrast and colour temperature are exactly the nuisance
    variables the three reported failures share (rubbered-in tarmac,
    overcast, floodlit night). Making the model invariant to them is a
    different move from CLAHE, which removed the cue at inference time and
    measurably cost 7 points of wet-vs-not-wet.

    No flips, no rotations: track surface has a direction (the racing line
    runs away from the camera) and there is no reason to teach the model
    that a mirrored world is the same world.
    """
    out = ImageEnhance.Brightness(img).enhance(rng.uniform(0.65, 1.35))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.7, 1.3))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.7, 1.3))
    return out


def centre_by_venue(X, races):
    """Subtract each venue's own mean embedding, then renormalise.

    Uses only the images of a venue, never their labels - information a
    deployed camera has for free before anyone says what the conditions are.
    """
    out = X.copy()
    for r in set(races):
        m = races == r
        out[m] = X[m] - X[m].mean(axis=0, keepdims=True)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-8, None)


def loro(X_train_src, X_test_src, y, races, aug_reps=0, X_aug=None):
    """Leave-one-race-out predictions.

    X_train_src and X_test_src are separate so a variant can train on
    augmented features while being tested on clean ones - which is the only
    honest way to measure augmentation.
    """
    pred = np.full(len(y), -1)
    for r in sorted(set(races)):
        te, tr = races == r, races != r
        if len(set(y[tr])) < 2:
            continue
        Xtr, ytr = X_train_src[tr], y[tr]
        if aug_reps and X_aug is not None:
            # X_aug is (reps, n, d); stack the training rows of each rep.
            extra = np.concatenate([X_aug[k][tr] for k in range(aug_reps)])
            Xtr = np.concatenate([Xtr, extra])
            ytr = np.concatenate([ytr] + [y[tr]] * aug_reps)
        clf = LogisticRegression(C=3.0, max_iter=3000,
                                 class_weight="balanced")
        clf.fit(Xtr, ytr)
        pred[te] = clf.predict(X_test_src[te])
    return pred


def report(name, pred, y, rows):
    ok = pred >= 0
    acc = float((pred[ok] == y[ok]).mean())
    wn = float(((pred[ok] == 2) == (y[ok] == 2)).mean())
    dry = ok & (y == 0)
    damp = ok & (y == 1)
    d_ok = float((pred[dry] == 0).mean()) if dry.any() else float("nan")
    m_ok = float((pred[damp] == 1).mean()) if damp.any() else float("nan")
    share = np.array([(pred[ok] == k).mean() for k in range(3)])
    flag = "  COLLAPSED" if share.max() >= 0.90 or share.min() == 0 else ""
    rows.append((name, acc, wn, d_ok, m_ok, flag))
    print(f"  {name:<22}{acc:>8.3f}{wn:>9.3f}{d_ok:>9.1%}{m_ok:>9.1%}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("calibrate"))
    ap.add_argument("--aug-reps", type=int, default=3,
                    help="augmented copies per training image (default 3)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="variant names to skip, e.g. dinov2")
    ap.add_argument("--seed", type=int, default=0)
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
    multi = sorted({r for r in set(races) if len(set(y[races == r])) > 1})
    print(f"{len(items)} frames  " + "  ".join(
        f"{c}={int((y == i).sum())}" for i, c in enumerate(CLASSES)))
    print(f"{len(set(races))} races, {len(multi)} multi-class: "
          f"{', '.join(multi) if multi else 'NONE'}")
    if not multi:
        print("\n!! No venue holds more than one class. Every variant below "
              "will be measuring venue identity - fix the data first.")

    rng = np.random.RandomState(args.seed)
    imgs = [Image.open(p).convert("RGB") for p, _ in items]

    # ---- CLIP, plain ----
    from backend.app import config as cfg
    from backend.app.models import clip_scorer as cs

    print("\nembedding: clip (plain)...")
    was = cfg.USE_CLAHE
    cfg.USE_CLAHE = cs.USE_CLAHE = False
    scorer = cs.ClipScorer()
    X = np.stack([scorer.embedding(im) for im in imgs])

    # ---- CLIP with CLAHE ----
    X_clahe = None
    if "clahe" not in args.skip:
        print("embedding: clip + clahe...")
        X_clahe = np.stack([scorer.embedding(cs.apply_clahe(im))
                            for im in imgs])

    # ---- CLIP on augmented copies (used for TRAINING only) ----
    X_aug = None
    if "augment" not in args.skip and args.aug_reps > 0:
        print(f"embedding: clip + augment "
              f"({args.aug_reps} copies x {len(imgs)})...")
        X_aug = np.stack([
            np.stack([scorer.embedding(augment(im, rng)) for im in imgs])
            for _ in range(args.aug_reps)])

    cfg.USE_CLAHE = cs.USE_CLAHE = was

    # ---- classical CV features, concatenated ----
    X_cv = None
    if "cv" not in args.skip:
        try:
            from backend.app.models.cv_features import extract as extract_cv
            print("computing: classical cv features...")
            feats = []
            for im in imgs:
                arr = np.array(im)
                mask = np.ones(arr.shape[:2], dtype=bool)
                d = extract_cv(arr, mask, "trackside")
                raw = d["raw"]
                feats.append([d["specular"] / 100.0, d["texture"] / 100.0,
                              d["spray"] / 100.0,
                              raw["specular_ratio"],
                              min(raw["laplacian_var"], 2000.0) / 2000.0,
                              raw["mean_brightness"] / 100.0])
            # Scaled into the same rough range as an L2-normalised
            # embedding, so the 512 CLIP dims do not simply drown them.
            X_cv = np.concatenate([X, np.array(feats) * 0.2], axis=1)
        except Exception as exc:
            print(f"  cv features unavailable: {type(exc).__name__}: {exc}")

    # ---- DINOv2 ----
    X_dino = None
    if "dinov2" not in args.skip:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            print("embedding: dinov2-small...")
            proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            mdl = AutoModel.from_pretrained("facebook/dinov2-small").eval()
            out = []
            with torch.no_grad():
                for im in imgs:
                    o = mdl(**proc(images=im, return_tensors="pt"))
                    e = getattr(o, "pooler_output", None)
                    if e is None:
                        e = o.last_hidden_state[:, 0]
                    e = e / e.norm(dim=-1, keepdim=True)
                    out.append(e.squeeze(0).numpy())
            X_dino = np.stack(out)
        except Exception as exc:
            print(f"  dinov2 unavailable: {type(exc).__name__}: {exc}")

    # ---- run them all ----
    print(f"\n{'variant':<24}{'3-class':>8}{'wet/not':>9}"
          f"{'dry-ok':>9}{'damp-ok':>9}")
    print("-" * 62)
    rows = []

    report("clip (baseline)", loro(X, X, y, races), y, rows)

    Xc = centre_by_venue(X, races)
    report("clip + centred", loro(Xc, Xc, y, races), y, rows)

    if X_clahe is not None:
        report("clip + clahe", loro(X_clahe, X_clahe, y, races), y, rows)

    if X_aug is not None:
        # Trained on clean + augmented, TESTED ON CLEAN.
        report("clip + augment",
               loro(X, X, y, races, args.aug_reps, X_aug), y, rows)
        Xac = np.stack([centre_by_venue(X_aug[k], races)
                        for k in range(args.aug_reps)])
        report("clip + aug + centred",
               loro(Xc, Xc, y, races, args.aug_reps, Xac), y, rows)

    if X_cv is not None:
        report("clip + cv features", loro(X_cv, X_cv, y, races), y, rows)

    if X_dino is not None:
        report("dinov2", loro(X_dino, X_dino, y, races), y, rows)
        Xd = centre_by_venue(X_dino, races)
        report("dinov2 + centred", loro(Xd, Xd, y, races), y, rows)

    base = rows[0][1]
    best = max(rows, key=lambda r: r[1])
    print("-" * 62)
    print(f"\nbaseline {base:.3f}   best '{best[0]}' {best[1]:.3f}   "
          f"{best[1] - base:+.3f}")
    if best[5]:
        print("  ...but the best variant COLLAPSED - a class it never emits")
        print("  cannot be detected in the field. Discount it.")
    elif best[1] - base < 0.02:
        print("  Nothing moved meaningfully. On THIS data the shipped")
        print("  baseline is already extracting what these features hold;")
        print("  the remaining errors sit at the dry/damp boundary, which is")
        print("  a genuine continuum rather than a shortcut.")
    else:
        print(f"  '{best[0]}' is worth adopting. Re-run train_probe.py with")
        print("  that change and confirm on validate_loro.py before shipping.")
    print("\nNOTHING WAS WRITTEN.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())