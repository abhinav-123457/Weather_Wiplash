#!/usr/bin/env python3
"""
extract_frames.py - pull labelled training frames out of race footage.

WHY THIS EXISTS
---------------
Hand-screenshotting produced 96 frames in roughly 16 hours, and those frames
still could not answer the question the project needs answered, because
NO VENUE IN THEM CONTAINS BOTH DRY AND DAMP. Measured consequence: under
leave-one-race-out, 29 of 31 dry frames are called damp - identically by a
linear head, two MLPs and an RBF kernel. "Is this damp" and "is this Monaco"
are the same question in that data.

So this tool is built around one goal: GET SEVERAL CONDITIONS OUT OF ONE
VENUE, quickly, without hand-labelling every image.

WHAT "CORRECTLY" MEANS HERE - four things, each learned the hard way
-------------------------------------------------------------------
1 VENUE-PREFIXED NAMES         tools/validate_loro.py splits on the leading
                               token, so monaco2024_04m12s.png groups with
                               its own race. Careless names silently break
                               every honest number in the project.

2 NEAR-DUPLICATE REJECTION     consecutive broadcast frames are almost the
                               same picture. Twenty of those are ONE sample
                               with nineteen copies, and a random train/test
                               split over them reported 83% where the true
                               figure was 36.5%. Frames are kept only if
                               they differ enough from what is already kept.

3 NON-TRACK REJECTION          broadcast cuts to crowd, garage, podium and
                               replay graphics constantly. Those frames are
                               not track surface and must never be labelled
                               as track. With --verify, CLIP checks each
                               candidate and discards anything that looks
                               like a cockpit, a grandstand or an overlay.

4 TIME RANGES                  a race is dry early and wet later. Running
                               this twice on ONE video with different
                               --start/--end and --label is how a single
                               venue produces several classes - the exact
                               thing the dataset is missing.

CAMERA TYPE - AND WHY IT IS A CONFOUND WAITING TO HAPPEN
--------------------------------------------------------
Race footage cuts between ONBOARD and TRACKSIDE shots, and they look
completely different. If dry frames end up mostly trackside and wet frames
mostly onboard, then "is this onboard" predicts the label exactly as well
as "is this Monaco" already does - the same failure, new variable.

So the camera type is detected per frame (CLIP), written into the filename,
and a BALANCE REPORT is printed at the end. Aim for both camera types in
every class. The report says so if you are drifting.

Detection also fixes a real trap: verifying the middle band of an ONBOARD
frame sees the halo and bodywork, not track, so a naive --verify would
discard every onboard frame. The band that gets checked depends on which
camera the shot came from - mirroring ROI_PRESETS in config.py.

USAGE
-----
    # dry phase of the race
    python tools/extract_frames.py monaco2023.mp4 --venue monaco2023 \\
        --label dry --start 0 --end 240 --every 4 --max 30 --verify

    # wet phase of the SAME video, same venue, different label
    python tools/extract_frames.py monaco2023.mp4 --venue monaco2023 \\
        --label wet --start 900 --end 1500 --every 4 --max 30 --verify

Writes to calibrate/<label>/. Use --dry-run first to see what it would keep
without writing anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv required:  pip install opencv-python-headless")

CLASSES = ("dry", "damp", "wet")

# Which band of the frame actually contains track surface, per camera type.
# Mirrors ROI_PRESETS in backend/app/config.py: an onboard shot is mostly
# car below the midline, a trackside shot puts the surface across the middle.
BANDS = {
    "onboard":   (0.10, 0.14, 0.90, 0.52),
    "trackside": (0.05, 0.25, 0.95, 0.88),
}

CAMERA_PROMPTS = {
    "onboard": [
        "view from inside a formula 1 car cockpit, halo bar and mirrors visible",
        "onboard camera on a racing car looking forward along the track",
        "the driver's view over the nose of a formula 1 car",
    ],
    "trackside": [
        "a television camera view of a racetrack from the side",
        "a whole racing car seen from beside the track",
        "wide view of a circuit with barriers, kerbs and grandstands",
    ],
}


class CameraClassifier:
    """Onboard vs trackside, using the CLIP already loaded for verification.

    Prompt-based rather than trained: this is a coarse, high-contrast choice
    between two visually unmistakable framings, which is the kind of call
    CLIP is dependable at. It is not the fine wet/damp distinction that
    needed a fitted probe.
    """

    def __init__(self, scorer):
        import torch
        from backend.app.models.clip_scorer import _as_embedding
        self.scorer = scorer
        self.names = list(CAMERA_PROMPTS)
        embs = []
        with torch.no_grad():
            for n in self.names:
                toks = scorer.processor(text=CAMERA_PROMPTS[n],
                                        return_tensors="pt", padding=True)
                e = _as_embedding(scorer.model.get_text_features(**toks),
                                  scorer.model, "text")
                e = e / e.norm(dim=-1, keepdim=True)
                m = e.mean(dim=0)
                embs.append((m / m.norm()).numpy())
        self.embs = np.stack(embs)

    def classify(self, pil_image) -> str:
        emb = self.scorer.embedding(pil_image)
        return self.names[int(np.argmax(emb @ self.embs.T))]


def signature(bgr) -> np.ndarray:
    """A tiny normalised fingerprint of a frame, for duplicate detection.

    32x32 greyscale, mean-subtracted and scaled to unit norm. Brightness and
    contrast drop out, so two views of the same corner still match even if
    the exposure drifted - which is what makes this catch real duplicates
    rather than just identical pixels.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32)).astype(np.float32).ravel()
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 1e-6 else g


def too_similar(sig, kept, threshold) -> bool:
    """Cosine similarity against every frame already kept."""
    return any(float(np.dot(sig, k)) >= threshold for k in kept)


def survey(args) -> int:
    """Dump timestamped thumbnails across the whole video.

    Deciding where a race is dry, damp or wet is a judgement about water,
    and the classifier being fixed is precisely the thing that cannot make
    it. So this makes no judgement at all - it lays the race out as a
    contact sheet and lets a person read it in one scroll, which beats
    scrubbing three videos by hand.
    """
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps

    step = max(args.every, 10.0)     # a survey wants coverage, not density
    dest = Path("survey") / args.venue
    dest.mkdir(parents=True, exist_ok=True)

    print(f"{args.video.name}  {duration/60:.1f} min")
    print(f"thumbnail every {step:.0f}s -> {dest}/")

    t, n = 0.0, 0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            thumb = cv2.resize(frame, (480, int(480 * h / w)))
            # Timestamp burned in, so the filename and the picture agree
            # even after the folder is sorted or shared.
            stamp = f"{int(t // 60):02d}m{int(t % 60):02d}s"
            cv2.putText(thumb, stamp, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(thumb, stamp, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(dest / f"{stamp}.jpg"), thumb)
            n += 1
        t += step
    cap.release()

    print(f"\n{n} thumbnails written.")
    print(f"""
NEXT
  Open {dest}/ and sort by name. Scroll once and note the timestamps where
  the track changes - rain arriving, spray appearing, a dry line forming.
  Then extract each phase with --start/--end:

      python tools/extract_frames.py {args.video.name} --venue {args.venue} \\
          --label dry --start 0 --end 300 --every 4 --max 30 --verify
""")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract diverse, verified training frames from a video.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--venue", required=True,
                    help="race identity, e.g. monaco2024. This becomes the "
                         "filename prefix and is what leave-one-race-out "
                         "holds out - keep it consistent per circuit+year.")
    # Not required at the parser level: --survey makes no judgement about
    # conditions, so demanding a label there would be asking for the very
    # answer the survey exists to help you find. Checked below instead.
    ap.add_argument("--label", choices=CLASSES,
                    help="required unless --survey")
    ap.add_argument("--every", type=float, default=4.0,
                    help="seconds between candidate frames (default 4)")
    ap.add_argument("--start", type=float, default=0.0, help="seconds")
    ap.add_argument("--end", type=float, default=None, help="seconds")
    ap.add_argument("--max", type=int, default=40, help="frames to keep")
    ap.add_argument("--similarity", type=float, default=0.92,
                    help="reject a frame this similar to one already kept "
                         "(0-1, lower = stricter, default 0.92)")
    ap.add_argument("--verify", action="store_true",
                    help="use CLIP to discard non-track shots (crowd, "
                         "garage, graphics). Slower, far cleaner.")
    ap.add_argument("--min-track", type=float, default=0.15,
                    help="with --verify, minimum 'track' probability to "
                         "keep a frame (default 0.15). Raise toward 0.4 to "
                         "be pickier, lower to 0.05 for street circuits "
                         "where barriers dominate the view.")
    ap.add_argument("--camera", choices=("auto", "onboard", "trackside"),
                    default="auto",
                    help="camera type for the filename. 'auto' detects it "
                         "per frame with CLIP (needs --verify) and reports "
                         "the balance, which is what stops camera type "
                         "becoming a second confound.")
    ap.add_argument("--out", type=Path, default=Path("calibrate"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be kept; write nothing")
    ap.add_argument("--survey", action="store_true",
                    help="STEP 1. Dump small timestamped thumbnails across "
                         "the whole video to survey/<venue>/ so you can see "
                         "at a glance where it is dry, damp and wet. Labels "
                         "nothing - it just tells you which --start/--end to "
                         "use. Ignores --label.")
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"no such video: {args.video}")

    if args.survey:
        return survey(args)

    if not args.label:
        ap.error("--label is required when not running --survey "
                 "(choose from: " + ", ".join(CLASSES) + ")")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if fps else 0
    end = args.end if args.end is not None else duration

    print(f"{args.video.name}  {duration/60:.1f} min "
          f"({duration:.0f}s) @ {fps:.0f} fps")

    # Seeking past the end returns unreadable frames forever, which looks
    # like a broken video rather than a bad argument. Say which it is.
    if duration and args.start >= duration:
        sys.exit(f"\n--start {args.start:.0f}s is past the end of this "
                 f"{duration:.0f}s video. Pick a range inside "
                 f"0-{duration:.0f}s (run with --survey to see where the "
                 f"conditions change).")
    if duration and end > duration:
        print(f"  --end {end:.0f}s trimmed to {duration:.0f}s (video length)")
        end = duration

    print(f"scanning {args.start:.0f}s -> {end:.0f}s every {args.every}s, "
          f"keeping at most {args.max} as '{args.label}' for '{args.venue}'")

    verifier = camera_clf = None
    if args.verify:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from backend.app.models.clip_scorer import ClipScorer
        print("\nloading CLIP for shot verification...")
        verifier = ClipScorer()
        if args.camera == "auto":
            camera_clf = CameraClassifier(verifier)
            print("  camera type will be detected per frame")
    elif args.camera == "auto":
        # Without CLIP there is nothing to detect with. Say so rather than
        # silently writing every frame under one camera tag.
        print("\n--camera auto needs --verify (it uses the same CLIP). "
              "Falling back to 'trackside' in filenames; pass --camera "
              "explicitly if that is wrong.")
        args.camera = "trackside"

    dest = args.out / args.label
    if not args.dry_run:
        dest.mkdir(parents=True, exist_ok=True)

    kept_sigs: list[np.ndarray] = []
    kept = 0
    by_camera = {"onboard": 0, "trackside": 0}
    dropped = {"duplicate": 0, "not_track": 0, "unreadable": 0}
    t = args.start

    while t < end and kept < args.max:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        t += args.every
        if not ok or frame is None:
            dropped["unreadable"] += 1
            continue

        sig = signature(frame)
        if too_similar(sig, kept_sigs, args.similarity):
            dropped["duplicate"] += 1
            continue

        camera = args.camera
        if verifier is not None:
            from PIL import Image
            h, w = frame.shape[:2]
            full = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if camera_clf is not None:
                camera = camera_clf.classify(full)

            # Check the band that camera type actually puts track surface
            # in. Using one fixed band would test the halo on onboard shots
            # and reject every one of them as "car".
            x0, y0, x1, y1 = BANDS[camera]
            band = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            rgb = Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB))
            v = verifier.verify_crop(rgb)
            p = v.get("probabilities", {})

            # Reject only what is definitely NOT usable track: a cockpit
            # close-up or a TV overlay. 'surroundings' is deliberately NOT
            # grounds for rejection.
            #
            # Measured: on Monaco this filter threw away 65 of 70 frames.
            # Monaco is a street circuit walled in by barriers and
            # buildings, so a perfectly good trackside shot scores highest
            # on 'surroundings' while still being most asphalt - the same
            # reason SegFormer reported "building 65%" on these frames at
            # the start of the project. Requiring 'track' to WIN discards
            # exactly the venue we most need frames from.
            bad = max(p.get("car", 0.0), p.get("graphics", 0.0))
            if p and (p.get("track", 0.0) < args.min_track or
                      bad > p.get("track", 0.0)):
                dropped["not_track"] += 1
                continue
            if not p and not v["is_track"]:
                dropped["not_track"] += 1        # older scorer, no probs
                continue

        name = (f"{args.venue}_{camera}_"
                f"{int(t // 60):02d}m{int(t % 60):02d}s.png")
        if not args.dry_run:
            cv2.imwrite(str(dest / name), frame)
        kept_sigs.append(sig)
        by_camera[camera] += 1
        kept += 1
        print(f"  kept {kept:>3}  {name}")

    cap.release()

    print(f"\nkept {kept}   dropped: " +
          "  ".join(f"{k}={v}" for k, v in dropped.items()))

    # ---- camera balance, across everything collected so far ----
    # Counted from DISK, not just this run, because the confound is a
    # property of the whole dataset - not of one invocation.
    print("\ncamera balance in calibrate/ (all runs)")
    print(f"  {'class':<8}{'onboard':>9}{'trackside':>11}")
    skewed = []
    for cls in CLASSES:
        d = args.out / cls
        if not d.is_dir():
            continue
        on = len(list(d.glob("*_onboard_*")))
        tr = len(list(d.glob("*_trackside_*")))
        other = len([p for p in d.iterdir()
                     if p.is_file() and "_onboard_" not in p.name
                     and "_trackside_" not in p.name])
        print(f"  {cls:<8}{on:>9}{tr:>11}"
              + (f"   (+{other} untagged)" if other else ""))
        if on + tr >= 8 and min(on, tr) == 0:
            skewed.append(cls)
    if skewed:
        print(f"\n  !! {skewed} have only ONE camera type.")
        print("     Camera type then predicts the class as reliably as the")
        print("     track does, and the model will learn that instead.")
        print("     Pull frames of the missing type before retraining.")
    else:
        print("\n  both camera types present where it matters.")
    if args.dry_run:
        print("DRY RUN - nothing was written.")
    else:
        print(f"written to {dest}/")

    if kept < args.max:
        print(f"\nOnly {kept} of {args.max} requested. Widen --start/--end, "
              f"lower --every, or raise --similarity toward 0.97 to accept "
              f"more similar frames (at the cost of diversity).")

    print(f"""
NEXT
  Run this AGAIN on the same video with a different --start/--end and
  --label, so ONE venue produces more than one condition. That is the
  single thing this dataset has never had, and it is what makes the
  dry-vs-damp distinction measurable at all:

      python tools/train_probe.py       # refit on the new frames
      python tools/validate_loro.py     # the honest, venue-held-out number
      python tools/test_capacity.py     # did dry-kept-dry move off 3.2%?
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())