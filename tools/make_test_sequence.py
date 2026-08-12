#!/usr/bin/env python3
"""
make_test_sequence.py - build an ordered sequence from the calibration set
so the temporal layer can be tested without a video.

THIS IS A TEST FIXTURE, NOT DEMO MATERIAL.
------------------------------------------
The trend it produces comes from the ORDER THE FILES WERE COPIED, not from
anything that happened on a track. It is perfect for answering "does the
smoothing, slope, hysteresis and DRYING derivation work end to end" and
useless for answering "does this system detect a real track drying".

Never show a sequence built by this script to a judge. If asked "are these
consecutive frames from one session?", the honest answer would be no - and
that question is an easy one to ask.

For the actual demo you need frames pulled from ONE continuous video across
a genuine transition. See extract_frames.py.

    python tools\\make_test_sequence.py --pattern drying
    python tools\\make_test_sequence.py --pattern storm --per-stage 4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Each pattern is the order of class folders to draw from.
PATTERNS = {
    "drying":  ["wet", "wet", "damp", "damp", "dry"],
    "wetting": ["dry", "dry", "damp", "damp", "wet"],
    "storm":   ["dry", "damp", "wet", "wet", "damp", "dry"],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("calibrate"))
    ap.add_argument("--out", type=Path, default=Path("test_sequence"))
    ap.add_argument("--pattern", choices=list(PATTERNS), default="drying")
    ap.add_argument("--per-stage", type=int, default=3,
                    help="frames drawn per stage (default 3)")
    ap.add_argument("--prefer", default=None,
                    help="prefer files whose name contains this, e.g. monaco")
    args = ap.parse_args()

    pools: dict[str, list[Path]] = {}
    for cls in ("dry", "damp", "wet"):
        folder = args.dir / cls
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                       ) if folder.is_dir() else []
        # Prefer one race where possible - frames from a single circuit keep
        # lighting and viewpoint roughly constant, so the score moves because
        # of wetness rather than because the whole scene changed.
        if args.prefer:
            pref = [p for p in files if args.prefer.lower() in p.name.lower()]
            files = pref + [p for p in files if p not in pref]
        pools[cls] = files
        if not files:
            print(f"! no images in {folder}")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    used: set[Path] = set()
    n = 0
    for stage in PATTERNS[args.pattern]:
        picked = 0
        for src in pools.get(stage, []):
            if src in used:
                continue
            n += 1
            dst = args.out / f"{n:03d}_{stage}_{src.name}"
            shutil.copy2(src, dst)
            used.add(src)
            picked += 1
            if picked >= args.per_stage:
                break
        if picked < args.per_stage:
            print(f"! only {picked} available for stage '{stage}'")

    print(f"\n{n} frames -> {args.out}/   pattern={args.pattern}")
    print("\nFilenames are numbered so they sort into the intended order.")
    print("\nRun it:")
    print(f"  python tools\\run_sequence.py {args.out}\\ --camera trackside")
    print("\nREMINDER: fixture only. The trend is the copy order, not a")
    print("real track drying. Do not demo this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
