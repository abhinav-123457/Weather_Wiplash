#!/usr/bin/env python3
"""
verify_lora.py - is the LoRA adapter actually running in the REAL app path?

An offline notebook number and working software are different things. This
loads the production ClipScorer exactly as the server does and checks four
things that the notebook cannot:

  1 does the adapter load at all, or did it silently fall back?
  2 does it actually change the embeddings, or is it a no-op?
  3 do the text-comparison paths still use the FROZEN tower?
    (LoRA adapts vision only; comparing an adapted image embedding against
     a frozen text embedding returns a number that means nothing)
  4 what does it cost per frame?

THIS IS AN INTEGRATION CHECK, NOT AN ACCURACY MEASUREMENT. It scores
frames the probe was trained on, so the accuracy printed at the end is
inflated by construction and is useful only for spotting gross breakage -
a healthy number here means "wired up", never "this good in the field".
The honest figure is the leave-one-race-out one from the notebook.

    python tools/verify_lora.py
    python tools/verify_lora.py --n 40
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASSES = ["dry", "damp", "wet"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("calibrate"))
    ap.add_argument("--n", type=int, default=30,
                    help="frames to score (default 30)")
    args = ap.parse_args()

    from backend.app import config as cfg
    from backend.app.models.clip_scorer import ClipScorer

    print(f"config: USE_LORA={cfg.USE_LORA}  LORA_PATH={cfg.LORA_PATH}  "
          f"USE_CLAHE={cfg.USE_CLAHE}")
    adapter = Path("backend/app") / cfg.LORA_PATH
    print(f"adapter dir exists: {adapter.exists()}  ({adapter})")
    if adapter.exists():
        files = sorted(p.name for p in adapter.iterdir())
        print(f"  contains: {', '.join(files[:6])}")
        if not any("adapter" in f for f in files):
            print("  !! no adapter_* files here. Unzip f1_lora.zip so that")
            print("     adapter_config.json sits directly in this folder.")

    print("\nloading the production ClipScorer...")
    scorer = ClipScorer()

    # ---- 1 & 2: is it live, and does it change anything? ----
    active = scorer.lora_vision is not None
    print(f"\n1. adapter active in ClipScorer: {active}")
    if cfg.USE_LORA and not active:
        print("   USE_LORA is on but the tower is frozen - read the startup")
        print("   lines above, they name the reason.")

    items = []
    for ci, cls in enumerate(CLASSES):
        d = args.dir / cls
        if d.is_dir():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    items.append((p, ci))
    if not items:
        sys.exit(f"no images under {args.dir}/[dry|damp|wet]/")

    rng = np.random.RandomState(0)
    pick = [items[i] for i in rng.choice(len(items),
                                         min(args.n, len(items)),
                                         replace=False)]
    imgs = [Image.open(p).convert("RGB") for p, _ in pick]

    e_live = np.stack([scorer.embedding(im) for im in imgs])
    e_froz = np.stack([scorer.embedding_frozen(im) for im in imgs])
    cos = float(np.mean(np.sum(e_live * e_froz, axis=1)))
    print(f"2. mean cosine(adapted, frozen) over {len(imgs)} frames: {cos:.4f}")
    if active and cos > 0.999:
        print("   !! essentially identical - the adapter is not doing")
        print("      anything. Treat any gain as unproven.")
    elif active:
        print("   embeddings genuinely differ, as they should")

    # ---- 3: text paths must be unaffected ----
    # A wrong answer here is invisible in normal use: scene context and the
    # crop verifier would just start returning slightly wrong labels.
    ctx = scorer.score_context(imgs[0])
    v = scorer.verify_crop(imgs[0])
    print(f"\n3. text-comparison paths (must use the FROZEN tower)")
    print(f"   scene   : {ctx.get('scene')} ({ctx.get('scene_confidence')})")
    print(f"   verifier: {v['kind']} ({v['confidence']})")
    print("   sane values here mean vision-text alignment survived; "
          "gibberish would mean it did not")

    # ---- 4: cost, and a gross-breakage check ----
    t0 = time.perf_counter()
    for im in imgs:
        scorer.embedding(im)
    per = (time.perf_counter() - t0) / len(imgs)
    print(f"\n4. {per*1000:.0f} ms per frame "
          f"({'adapted' if active else 'frozen'} tower)")

    if scorer.probe is not None:
        y = np.array([c for _, c in pick])
        pred = np.array([CLASSES.index(scorer.probe_from(e)[3])
                         for e in e_live])
        acc = float((pred == y).mean())
        share = [float((pred == k).mean()) for k in range(3)]
        print(f"\n   TRAIN-SET accuracy {acc:.3f}  "
              f"(inflated - these frames trained the probe)")
        print("   prediction spread: " + "  ".join(
            f"{c} {share[k]:.0%}" for k, c in enumerate(CLASSES)))
        if min(share) == 0:
            print("   !! a class is never predicted - something is wrong")
            print("      with the head/embedding pairing. Do not ship.")
        elif acc < 0.6:
            print("   !! low even on training data - the head and the")
            print("      embeddings probably come from different runs.")
        else:
            print("   looks wired up correctly")

    print("""
IF ALL FOUR LOOK RIGHT
  Start the backend and run the demo end to end - especially the footage
  that used to fail. That is the only test that counts.
  Reverting is one line: USE_LORA = False in backend/app/config.py.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
