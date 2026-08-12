#!/usr/bin/env python3
"""
run_sequence.py - push a folder of frames through the API in order and watch
the trend build.

This is how you test the temporal layer. Uploading twelve images by hand
through /docs and comparing JSON blobs tells you almost nothing; the whole
point is the SHAPE of the sequence, and that only shows up when the frames
are seen together.

    # start the backend first, then:
    python tools\\run_sequence.py demo_frames\\
    python tools\\run_sequence.py demo_frames\\ --camera onboard
    python tools\\run_sequence.py demo_frames\\ --roi 0.1,0.15,0.9,0.52

FRAME ORDER IS FILENAME ORDER, so name them so they sort correctly -
frame_001.jpg, frame_002.jpg. Frames pulled by extract_frames.py are named
by timestamp and already sort right.

The session is reset before the run, so repeated runs never contaminate
each other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx required (it ships with fastapi):\n"
             "    .\\.venv\\Scripts\\pip.exe install httpx")

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sparkline(values: list[float], labels: list[str],
              height: int = 14) -> str:
    """Vertical ASCII chart of the smoothed wetness track.

    Terminal-only, but it makes the trend legible long before any frontend
    exists - and the trend is the entire product.
    """
    if not values:
        return ""
    width = len(values)
    rows = []
    for r in range(height, -1, -1):
        lo = r * 100.0 / height
        hi = (r + 1) * 100.0 / height
        line = ""
        for v in values:
            line += "#" if lo <= v < hi or (r == height and v >= 100) else " "
        axis = f"{lo:5.0f} |"
        rows.append(axis + line)

    # Mark where the committed label changes - the moments that matter.
    marks = " " * 7
    prev = None
    for lab in labels:
        marks += ("|" if prev is not None and lab != prev else " ")
        prev = lab
    return "\n".join(rows) + "\n" + " " * 6 + "+" + "-" * width + \
           "\n" + marks + "  <- label changes"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--camera", default="trackside",
                    choices=["trackside", "onboard"])
    ap.add_argument("--roi", default=None,
                    help='"x0,y0,x1,y1" fractions; omit for the camera preset')
    ap.add_argument("--session", default="sequence")
    args = ap.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        sys.exit(f"no images in {args.folder}")

    client = httpx.Client(timeout=120.0)

    try:
        client.get(f"{args.url}/health").raise_for_status()
    except Exception as exc:
        sys.exit(f"backend not reachable at {args.url}\n  {exc}\n"
                 f"  start it: .\\.venv\\Scripts\\python.exe -m uvicorn "
                 f"backend.app.main:app --port 8000")

    # Always reset. Without this a second run appends to the first and the
    # trend line is nonsense.
    client.post(f"{args.url}/api/session/reset",
                data={"session_id": args.session})
    print(f"session '{args.session}' reset · {len(files)} frames · "
          f"camera={args.camera}\n")

    hdr = (f"{'#':>3}  {'file':<26}{'raw':>6}{'smooth':>8}{'slope':>7}  "
           f"{'state':<5} {'trend':<8} {'LABEL':<8}{'conf':>6}")
    print(hdr)
    print("-" * len(hdr))

    smooth_track: list[float] = []
    label_track: list[str] = []

    for path in files:
        with open(path, "rb") as fh:
            data = {"camera_type": args.camera, "session_id": args.session}
            if args.roi:
                data["roi"] = args.roi
            r = client.post(f"{args.url}/api/analyze/image",
                            files={"file": (path.name, fh, "image/jpeg")},
                            data=data)
        if r.status_code != 200:
            print(f"     {path.name:<26} HTTP {r.status_code}")
            continue

        j = r.json()
        if "error" in j:
            print(f"     {path.name:<26} {j['error']}: "
                  f"{j.get('message', '')[:40]}")
            continue

        smooth_track.append(j["wetness"])
        label_track.append(j["label"])

        # Flag the frames where the committed label actually moved - those
        # are the moments a race engineer would act on.
        changed = (len(label_track) > 1
                   and label_track[-1] != label_track[-2])
        print(f"{j['frame']:>3}  {path.name[:26]:<26}"
              f"{j['wetness_raw']:>6.1f}{j['wetness']:>8.1f}"
              f"{j['slope']:>7.2f}  {j['state']:<5} {j['trend']:<8} "
              f"{j['label']:<8}{j['confidence']:>6.2f}"
              + ("   <-- changed" if changed else ""))

    if smooth_track:
        print("\nsmoothed wetness\n")
        print(sparkline(smooth_track, label_track))
        print(f"\nstart {smooth_track[0]:.1f} -> end {smooth_track[-1]:.1f}"
              f"   net {smooth_track[-1] - smooth_track[0]:+.1f}")
        seen = []
        for lab in label_track:
            if not seen or seen[-1] != lab:
                seen.append(lab)
        print("label path: " + " -> ".join(seen))

    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
