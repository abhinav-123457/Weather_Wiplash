#!/usr/bin/env python3
"""
audit_dataset.py - is the training set actually able to answer the question?

Run this BEFORE train_probe.py. It loads no models and takes a second: it
only reads filenames, because every confound this project has hit was
visible in the filenames long before it cost a training run.

THE ONE QUESTION
----------------
Does any venue contain MORE THAN ONE class?

If not, "which class is this" and "which circuit is this" are the same
question, and no model can tell them apart. That was the measured state of
the original 96 frames: five of six races single-class, and 29 of 31 dry
frames called damp - identically by a linear head, two MLPs and an RBF
kernel, because the shortcut was in the labels rather than the model.

ALSO CHECKED
  camera balance   if dry is all trackside and wet all onboard, camera type
                   predicts the label just as well as venue did
  class balance    a class with a handful of frames cannot be learned
  naming           frames the venue parser cannot read are invisible to
                   leave-one-race-out, which silently corrupts the number

    python tools/audit_dataset.py
    python tools/audit_dataset.py --dir calibrate
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

CLASSES = ("dry", "damp", "wet")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def race_of(stem: str) -> str:
    """Must match validate_loro.py exactly, or the audit measures nothing."""
    m = re.match(r"([A-Za-z]+\d{4})", stem)
    return (m.group(1) if m else stem.split("_")[0]).lower()


def camera_of(name: str) -> str:
    if "_onboard_" in name:
        return "onboard"
    if "_trackside_" in name:
        return "trackside"
    return "untagged"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("calibrate"))
    args = ap.parse_args()

    by_venue = defaultdict(lambda: defaultdict(int))
    by_camera = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    unparsed = []

    for cls in CLASSES:
        d = args.dir / cls
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            venue = race_of(p.stem)
            by_venue[venue][cls] += 1
            by_camera[camera_of(p.name)][cls] += 1
            totals[cls] += 1
            # A stem with no letters+4-digits falls back to the first token,
            # which groups unrelated files together and quietly breaks LORO.
            if not re.match(r"[A-Za-z]+\d{4}", p.stem):
                unparsed.append(p.name)

    n = sum(totals.values())
    if not n:
        raise SystemExit(f"no images under {args.dir}/[dry|damp|wet]/")

    print(f"{n} frames  " + "  ".join(f"{c}={totals[c]}" for c in CLASSES))

    # ---- the decisive table ----
    print(f"\n{'venue':<20}" + "".join(f"{c:>7}" for c in CLASSES)
          + f"{'classes':>9}")
    print("-" * 56)
    multi = []
    for venue in sorted(by_venue):
        row = by_venue[venue]
        k = sum(1 for c in CLASSES if row[c] > 0)
        flag = "  <-- multi" if k > 1 else ""
        print(f"{venue:<20}" + "".join(f"{row[c]:>7}" for c in CLASSES)
              + f"{k:>9}{flag}")
        if k > 1:
            multi.append(venue)

    print(f"\n{'camera':<20}" + "".join(f"{c:>7}" for c in CLASSES))
    print("-" * 47)
    for cam in ("onboard", "trackside", "untagged"):
        if by_camera[cam]:
            print(f"{cam:<20}"
                  + "".join(f"{by_camera[cam][c]:>7}" for c in CLASSES))

    # ---- verdict ----
    print("\n" + "=" * 56)
    if not multi:
        print("CONFOUNDED. No venue holds more than one class, so 'which")
        print("class' and 'which venue' are the same question. Training on")
        print("this teaches venue identity - measured before at 29 of 31")
        print("dry frames called damp, unchanged across four model")
        print("capacities. Extract a second condition from a venue you")
        print("already have before going further.")
    else:
        print(f"USABLE. {len(multi)} venue(s) hold several classes: "
              f"{', '.join(multi)}")
        print("Within those, lighting and camera are held constant while the")
        print("water changes - which is the only setting where the")
        print("dry-vs-damp boundary can actually be measured. This is the")
        print("first time that has been true in this project.")

    # A venue needs both sides of a boundary AND enough frames to hold out.
    thin = [v for v in multi if sum(by_venue[v].values()) < 12]
    if thin:
        print(f"\n  note: {thin} have few frames. Leave-one-race-out holds "
              f"the whole venue out,\n  so a thin one gives a noisy number "
              f"rather than a wrong one.")

    small = [c for c in CLASSES if 0 < totals[c] < 20]
    if small:
        print(f"\n  note: {small} under 20 frames - expect unstable recall "
              f"for those.")

    if unparsed:
        print(f"\n  !! {len(unparsed)} file(s) whose venue cannot be parsed, "
              f"e.g. {unparsed[0]}")
        print("     LORO groups these by the first underscore token, which")
        print("     may merge unrelated races. Rename to <venue><year>_...")

    print(f"""
NEXT
  python tools/train_probe.py       refit on this data
  python tools/test_capacity.py     did dry-kept-dry move off 3.2%?
  python tools/validate_loro.py     the honest, venue-held-out number
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())