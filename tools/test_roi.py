#!/usr/bin/env python3
"""
test_roi.py - which ROI preset and squaring setting actually scores best?

THE MISMATCH THIS EXISTS TO MEASURE
-----------------------------------
The probe is TRAINED on whole frames: extract_frames.py writes the full
picture, and train_probe.py embeds it as-is. But at inference main.py crops
to the ROI first. So training and serving have never seen the same kind of
image, and every accuracy number so far was measured on whole frames while
the app runs on crops.

That makes ROI choice a real variable, not a cosmetic one - and the answer
is not obvious in either direction:

  a bigger ROI  matches training more closely, but scores sky, barriers,
                grandstands and car bodywork as if they were track
  a tighter ROI is honest about what it looks at, but hands the model a
                kind of picture it was never fitted on

CLIPProcessor also matters: it resizes the SHORTEST side to 224 then
centre-crops. So a whole 16:9 frame is effectively cropped to its middle
square before the model ever sees it, while ROI_TO_SQUARE=True stretches
the selection instead. Those are different pictures, so both are tested.

HOW
Each candidate is applied to EVERY frame - training and test alike - and
scored leave-one-race-out. Applying a crop only at test time would measure
the mismatch rather than the crop.

Camera type comes from the filename (_onboard_ / _trackside_), exactly as
the app applies its presets.

WRITES NOTHING.

    python tools/test_roi.py
    python tools/test_roi.py --quick     # skip the no-square variants
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    sys.exit("scikit-learn required")

CLASSES = ["dry", "damp", "wet"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
FULL = [0.0, 0.0, 1.0, 1.0]

# (name, {camera: box}) - box is x0,y0,x1,y1 in 0-1 fractions.
CANDIDATES = {
    # What ships today.
    "current": {"onboard": [0.08, 0.14, 0.92, 0.55],
                "trackside": [0.03, 0.20, 0.97, 0.90]},
    # The proposal: onboard gets the current trackside band, trackside
    # takes the whole frame.
    "proposed": {"onboard": [0.03, 0.20, 0.97, 0.90],
                 "trackside": FULL},
    # Everything, both cameras - the closest match to how the probe was
    # actually trained.
    "full-both": {"onboard": FULL, "trackside": FULL},
    # A wider onboard band that still stops above the cockpit.
    "wide-onboard": {"onboard": [0.02, 0.12, 0.98, 0.62],
                     "trackside": [0.02, 0.12, 0.98, 0.95]},
}


def race_of(path: Path) -> str:
    m = re.match(r"([A-Za-z]+\d{4})", path.stem)
    return (m.group(1) if m else path.stem.split("_")[0]).lower()


def camera_of(name: str) -> str:
    if "_onboard_" in name:
        return "onboard"
    if "_trackside_" in name:
        return "trackside"
    return "trackside"          # app default when nothing says otherwise


def prep(img: Image.Image, box, square: bool, size: int = 224):
    w, h = img.size
    crop = img.crop((int(box[0] * w), int(box[1] * h),
                     int(box[2] * w), int(box[3] * h)))
    return crop.resize((size, size)) if square else crop


def loro(X, y, races):
    pred = np.full(len(y), -1)
    for r in sorted(set(races)):
        te, tr = races == r, races != r
        if len(set(y[tr])) < 2:
            continue
        clf = LogisticRegression(C=3.0, max_iter=3000,
                                 class_weight="balanced")
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("calibrate"))
    ap.add_argument("--quick", action="store_true",
                    help="squared variants only")
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
    cams = np.array([camera_of(p.name) for p, _ in items])
    print(f"{len(items)} frames  " + "  ".join(
        f"{c}={int((y == i).sum())}" for i, c in enumerate(CLASSES)))
    print(f"cameras: " + "  ".join(
        f"{c}={int((cams == c).sum())}" for c in ("onboard", "trackside")))

    from backend.app import config as cfg
    from backend.app.models.clip_scorer import ClipScorer
    print(f"\nUSE_LORA={cfg.USE_LORA}  (whichever tower is live is the one "
          f"being measured)")
    scorer = ClipScorer()
    imgs = [Image.open(p).convert("RGB") for p, _ in items]

    squares = [True] if args.quick else [True, False]
    rows = []
    print(f"\n{'preset':<16}{'square':>8}{'3-class':>9}{'wet/not':>9}"
          f"{'dry-ok':>9}{'damp-ok':>9}{'frame %':>9}")
    print("-" * 69)

    for name, preset in CANDIDATES.items():
        for square in squares:
            X = np.stack([
                scorer.embedding(prep(im, preset[cams[i]], square))
                for i, im in enumerate(imgs)])
            pred = loro(X, y, races)
            ok = pred >= 0
            acc = float((pred[ok] == y[ok]).mean())
            wn = float(((pred[ok] == 2) == (y[ok] == 2)).mean())
            dry = ok & (y == 0)
            damp = ok & (y == 1)
            d_ok = float((pred[dry] == 0).mean()) if dry.any() else np.nan
            m_ok = float((pred[damp] == 1).mean()) if damp.any() else np.nan
            # Mean fraction of the frame each camera's box covers.
            area = np.mean([(preset[c][2] - preset[c][0]) *
                            (preset[c][3] - preset[c][1]) for c in cams])
            share = np.array([(pred[ok] == k).mean() for k in range(3)])
            flag = "  COLLAPSED" if share.max() >= 0.9 or share.min() == 0 else ""
            rows.append((name, square, acc, wn, d_ok, m_ok, flag))
            print(f"  {name:<14}{str(square):>8}{acc:>9.3f}{wn:>9.3f}"
                  f"{d_ok:>9.1%}{m_ok:>9.1%}{area*100:>8.0f}%{flag}")

    cur = next(r for r in rows if r[0] == "current" and r[1] is True)
    best = max(rows, key=lambda r: r[2])
    print("-" * 69)
    print(f"\ncurrent {cur[2]:.3f}   best '{best[0]}' square={best[1]} "
          f"{best[2]:.3f}   {best[2] - cur[2]:+.3f}")
    if best[6]:
        print("  ...but it COLLAPSED. Discount it.")
    elif best[2] - cur[2] < 0.03:
        print("  Within noise on this much data. ROI framing is not the")
        print("  bottleneck; keep the current presets, which at least look")
        print("  at track rather than sky and bodywork.")
    else:
        print(f"  Worth adopting. Set ROI_PRESETS in config.py to the")
        print(f"  '{best[0]}' boxes and ROI_TO_SQUARE = {best[1]}.")
        print("  Then re-run train_probe.py so the probe is fitted on the")
        print("  same crops the app will feed it - which is the mismatch")
        print("  this whole test exists to close.")
    print("\nNOTHING WAS WRITTEN.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())