#!/usr/bin/env python3
"""
smoke_test.py - Stage 0 verification.

Answers the four questions that could derail the build, before any real code
gets written:

  1. Does the environment import cleanly, with a CPU-only torch?
  2. Do both Hugging Face models download and load?
  3. Does transformers v5 still expose the APIs we plan to call?
  4. What is the ACTUAL per-frame time and memory cost on this machine?

Question 4 matters most. Every timing figure in the build plan is an estimate.
This replaces them with measurements from the machine you will demo on.

    python setup\\smoke_test.py          (Windows)
    python setup/smoke_test.py           (Linux/macOS)

Send the whole output back - the numbers drive the video frame budget.
"""

from __future__ import annotations

import platform
import sys
import time
import traceback

SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# State prompts anchored to wetness values. No "drying" prompt - drying is a
# trend, derived from slope across frames, never read from a single image.
CLIP_ANCHORS = [
    ("a completely dry asphalt racetrack", 0),
    ("a damp asphalt racetrack, dark surface, no standing water", 40),
    ("a wet asphalt racetrack with reflections and standing water", 85),
    ("a flooded racetrack with heavy standing water and spray", 100),
]

failures: list[str] = []


def head(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    failures.append(msg)


def note(msg: str) -> None:
    print(f"         {msg}")


def rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


# ---------------------------------------------------------------------------
head("1. ENVIRONMENT")
# ---------------------------------------------------------------------------
print(f"  Python    {sys.version.split()[0]}")
print(f"  Platform  {platform.system()} {platform.release()} ({platform.machine()})")
print(f"  Processor {platform.processor() or 'unknown'}")

try:
    import psutil
    print(f"  CPU       {psutil.cpu_count(logical=False)} cores / "
          f"{psutil.cpu_count(logical=True)} threads")
    vm = psutil.virtual_memory()
    print(f"  RAM       {vm.total / 1e9:.1f} GB total, "
          f"{vm.available / 1e9:.1f} GB available")
    if vm.available < 4e9:
        note("WARNING: under 4 GB free. Close browser tabs before the demo.")
except ImportError:
    note("psutil not installed - skipping CPU/RAM report")

try:
    import torch
    print(f"  torch     {torch.__version__}")
    if "+cpu" in torch.__version__ or not torch.cuda.is_available():
        ok("CPU-only build, as intended")
    else:
        note("CUDA build detected. Harmless, but it means ~2.5 GB of unused "
             "NVIDIA packages were installed.")
    torch.set_num_threads(psutil.cpu_count(logical=False) if "psutil" in sys.modules else 4)
except Exception as exc:
    bad(f"torch import failed: {exc}")
    sys.exit(1)

try:
    import transformers
    print(f"  transformers {transformers.__version__}")
    if not transformers.__version__.startswith("5"):
        note("NOT v5. The build plan targets v5 - APIs differ from v4.")
except Exception as exc:
    bad(f"transformers import failed: {exc}")
    sys.exit(1)

for mod in ("cv2", "PIL", "numpy", "fastapi"):
    try:
        __import__(mod)
        ok(f"{mod} imports")
    except Exception as exc:
        bad(f"{mod} import failed: {exc}")

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
head("2. SEGFORMER  (downloads ~15 MB on first run)")
# ---------------------------------------------------------------------------
seg_model = seg_proc = None
road_ids: list[int] = []

try:
    from transformers import SegformerForSemanticSegmentation, AutoImageProcessor

    t0 = time.perf_counter()
    seg_proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        SEG_MODEL, use_safetensors=True)
    seg_model.eval()
    ok(f"loaded in {time.perf_counter() - t0:.1f}s")

    # Never hardcode the class index - look it up. Label ordering is a
    # property of the checkpoint, not something to assume.
    id2label = seg_model.config.id2label
    road_ids = [i for i, l in id2label.items() if "road" in str(l).lower()]
    if road_ids:
        ok(f"road class ids: {road_ids}  "
           f"({', '.join(str(id2label[i]) for i in road_ids)})")
    else:
        bad("no 'road' class found in id2label - segmentation masking will fail")
        note(f"first 12 labels: {[str(id2label[i]) for i in sorted(id2label)[:12]]}")

except Exception as exc:
    bad(f"SegFormer failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# transformers v4 -> v5 compatibility
#
# In v4, get_text_features()/get_image_features() returned a plain tensor.
# In v5 they return a BaseModelOutputWithPooling, so `.norm()` blows up.
#
# Worse, pooler_output is NOT the embedding we want: CLIP's towers have
# different hidden sizes (text 512, vision 768) and only become comparable
# after their projection layers map them into the shared space. Using
# pooler_output directly would either crash on the matmul or, if the dims
# happened to line up, silently compute nonsense.
#
# This helper handles both versions and applies the projection when needed.
# ---------------------------------------------------------------------------
_paths_seen: set[str] = set()


def as_embedding(out, model, kind: str):
    """Normalise get_*_features output to a shared-space embedding tensor.

    Never assume projection is needed - CHECK THE DIMENSION. In v5,
    pooler_output already comes back projected to config.projection_dim.
    Projecting again raises for vision (768x512 against a 512 input) and,
    worse, silently succeeds for text on ViT-B/32 where the projection is
    512->512. A shape check would not catch that; only comparing against
    projection_dim does.
    """
    def seen(path: str):
        key = f"{kind}:{path}"
        if key not in _paths_seen:
            _paths_seen.add(key)
            note(f"{kind} embeddings via {path}")

    if torch.is_tensor(out):
        seen("tensor (transformers v4)")
        return out

    for attr in ("text_embeds", "image_embeds"):
        val = getattr(out, attr, None)
        if torch.is_tensor(val):
            seen(f"out.{attr}")
            return val

    pooled = getattr(out, "pooler_output", None)
    if torch.is_tensor(pooled):
        target = getattr(model.config, "projection_dim", None)
        if target is None or pooled.shape[-1] == target:
            seen(f"pooler_output (already {pooled.shape[-1]}-d, no projection)")
            return pooled
        proj = model.text_projection if kind == "text" else model.visual_projection
        seen(f"pooler_output {pooled.shape[-1]}-d -> projection -> {target}-d")
        return proj(pooled)

    raise TypeError(f"cannot extract embedding from {type(out).__name__}")


# ---------------------------------------------------------------------------
head("3. CLIP  (downloads ~600 MB on first run)")
# ---------------------------------------------------------------------------
clip_model = clip_proc = None
text_embs = None

try:
    from transformers import CLIPModel, CLIPProcessor

    t0 = time.perf_counter()
    clip_proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    # use_safetensors avoids pulling the legacy 605 MB pytorch_model.bin
    # alongside the safetensors copy - your first run downloaded both.
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True)
    clip_model.eval()
    ok(f"loaded in {time.perf_counter() - t0:.1f}s")

    # Encode the prompts ONCE. They never change, so the 63M-parameter text
    # tower should never run per frame. Biggest optimisation available.
    prompts = [p for p, _ in CLIP_ANCHORS]
    t0 = time.perf_counter()
    with torch.no_grad():
        toks = clip_proc(text=prompts, return_tensors="pt", padding=True)
        text_embs = as_embedding(clip_model.get_text_features(**toks),
                                 clip_model, "text")
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
    ok(f"cached {len(prompts)} text embeddings in "
       f"{time.perf_counter() - t0:.2f}s  shape={tuple(text_embs.shape)}")

except Exception as exc:
    bad(f"CLIP failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
head("4. TIMED PIPELINE  (the numbers that matter)")
# ---------------------------------------------------------------------------
if seg_model is None or clip_model is None:
    bad("skipping - a model failed to load")
else:
    # Synthetic road-ish frame: grey asphalt band under a lighter sky.
    # Correctness of the label is irrelevant here; we are measuring cost.
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[: h // 3] = (150, 165, 185)                       # sky
    frame[h // 3 :] = (95, 95, 98)                          # asphalt
    frame += np.random.randint(0, 14, frame.shape, dtype=np.uint8)
    image = Image.fromarray(frame)

    def run_once():
        """Returns (t_seg, t_clip, t_cv, sims)."""
        t0 = time.perf_counter()
        with torch.no_grad():
            seg_model(**seg_proc(images=image, return_tensors="pt"))
        t_seg = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            img_emb = as_embedding(
                clip_model.get_image_features(
                    **clip_proc(images=image, return_tensors="pt")),
                clip_model, "image")
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            # Loud failure beats a silent wrong answer: if the towers were
            # not projected into the shared space these dims will differ.
            assert img_emb.shape[-1] == text_embs.shape[-1], (
                f"embedding dim mismatch: image {img_emb.shape[-1]} vs "
                f"text {text_embs.shape[-1]} - projection not applied")
            sims = (img_emb @ text_embs.T).squeeze(0)
        t_clip = time.perf_counter() - t0

        t0 = time.perf_counter()
        import cv2
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _ = float(g.mean())
        hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
        _ = int((hls[:, :, 1] > 200).sum())
        t_cv = time.perf_counter() - t0

        return t_seg, t_clip, t_cv, sims

    print("  warming up...")
    run_once()

    runs = 5
    print(f"  timing {runs} frames at {w}x{h}...\n")
    seg_t, clip_t, cv_t = [], [], []
    sims = None
    for _ in range(runs):
        a, b, c, sims = run_once()
        seg_t.append(a); clip_t.append(b); cv_t.append(c)

    med = lambda xs: sorted(xs)[len(xs) // 2]
    m_seg, m_clip, m_cv = med(seg_t), med(clip_t), med(cv_t)
    total = m_seg + m_clip + m_cv

    print(f"    SegFormer   {m_seg * 1000:7.0f} ms")
    print(f"    CLIP image  {m_clip * 1000:7.0f} ms")
    print(f"    OpenCV      {m_cv * 1000:7.0f} ms")
    print(f"    {'-' * 26}")
    print(f"    TOTAL       {total * 1000:7.0f} ms  ({total:.2f} s/frame)\n")

    budget = max(1, int(30 / total))
    note(f"~{budget} frames fit in a 30-second processing window.")
    note(f"Suggested video frame budget: {min(budget, 50)}")

    if total < 0.6:
        ok("comfortably fast - SSE streaming will feel smooth")
    elif total < 1.5:
        ok("workable - matches the plan's 0.8-1.2 s/frame estimate")
    else:
        note("slower than planned. Reduce the video frame budget, and "
             "consider resizing frames to 512px before SegFormer.")

    r = rss_mb()
    if r:
        note(f"process memory with both models loaded: {r:.0f} MB")

    # Sanity check, not a correctness test: a uniform grey frame has no
    # ground truth. We only confirm the scorer produces a usable number.
    # Cosine similarities are the real health check. Correctly projected CLIP
    # embeddings land roughly in 0.15-0.35 against natural-language prompts.
    # A double-projected or unprojected vector still produces a valid-looking
    # score, but the similarities collapse toward 0 or drift far outside that
    # band - which a shape assertion can never detect.
    s = sims.numpy()
    print("\n  cosine similarity per prompt:")
    for (prompt, val), sim in zip(CLIP_ANCHORS, s):
        print(f"    {sim:+.3f}  [{val:>3}] {prompt[:52]}")

    if 0.10 <= float(s.mean()) <= 0.40:
        ok(f"similarities in the expected CLIP range (mean {s.mean():+.3f})")
    else:
        bad(f"similarities out of range (mean {s.mean():+.3f}) - embeddings "
            f"are likely mis-projected")

    probs = torch.softmax(sims / 0.01, dim=-1).numpy()
    wetness = float(probs @ np.array([v for _, v in CLIP_ANCHORS]))
    print(f"\n  synthetic frame wetness score: {wetness:.1f} / 100")
    note("The score itself is meaningless on a synthetic image - it only")
    note("proves the path runs. Real calibration comes from RoadSaW later.")


# ---------------------------------------------------------------------------
head("RESULT")
# ---------------------------------------------------------------------------
if failures:
    print(f"  {len(failures)} FAILURE(S):\n")
    for f in failures:
        print(f"    - {f}")
    print("\n  Send this whole output back before starting Stage 1.")
    sys.exit(1)

print("  All checks passed. Environment is ready for Stage 1.")
print("  Send the timing numbers back so the frame budget can be set.")
sys.exit(0)
