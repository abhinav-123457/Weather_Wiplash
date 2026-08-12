"""
segmentation.py - isolate the track surface.

Scoring a whole frame would let sky, grass, crowd and barriers pull the
wetness estimate around. Masking to road pixels first means CLIP and the
classical CV features only ever see asphalt.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from ..config import ROAD_LABEL_KEYWORDS, SEG_MODEL


def _runs(fracs, lo):
    """Contiguous runs where fracs >= lo, sorted by total mass, descending."""
    ok = fracs >= lo
    runs, start, mass = [], None, 0.0
    for i in range(len(ok)):
        if ok[i]:
            if start is None:
                start, mass = i, 0.0
            mass += float(fracs[i])
        if (not ok[i] or i == len(ok) - 1) and start is not None:
            end = i + 1 if ok[i] else i
            runs.append((start, end, mass))
            start = None
    return sorted(runs, key=lambda r: -r[2])


def road_box_candidates(mask: np.ndarray, k: int = 3,
                        min_row_frac: float = 0.20,
                        min_col_frac: float = 0.12,
                        pad: float = 0.02) -> list[list]:
    """Up to k density-trimmed candidate boxes, biggest road mass first.

    Why CANDIDATES and not one box: the Cityscapes model carries a strong
    bottom-of-frame prior - measured live, it labelled an onboard car's
    dark bodywork "road", so the single biggest region WAS the car. The
    segmenter cannot adjudicate its own mistake; it proposes, and the
    caller verifies each crop with CLIP (does this look like asphalt, or
    like a cockpit / TV graphics?) before any box is trusted.

    Each candidate: the contiguous band of road-heavy ROWS, then the
    strongest COLUMN run within it - always by road mass, never by length.
    Boxes under ~2% of the frame are dropped; CLIP cannot judge a sliver.
    """
    h, w = mask.shape
    out = []
    for y0, y1, _m in _runs(mask.mean(axis=1), min_row_frac)[:k]:
        cols = _runs(mask[y0:y1].mean(axis=0), min_col_frac)
        if not cols:
            continue
        x0, x1, _cm = cols[0]
        box = [max(0.0, x0 / w - pad), max(0.0, y0 / h - pad),
               min(1.0, x1 / w + pad), min(1.0, y1 / h + pad)]
        if (box[2] - box[0]) * (box[3] - box[1]) >= 0.02:
            out.append(box)
    return out


def road_box(mask: np.ndarray, **kw) -> list | None:
    """Single best candidate - kept for callers that cannot verify."""
    c = road_box_candidates(mask, k=1, **kw)
    return c[0] if c else None


class RoadSegmenter:
    """SegFormer-b0 wrapper that returns a binary road mask.

    Measured at 290ms/frame - about 73% of total pipeline cost, despite being
    23x smaller than CLIP. Dense per-pixel prediction is simply expensive.
    If you need speed, reduce the input size here rather than touching CLIP.
    """

    def __init__(self, model_name: str = SEG_MODEL) -> None:
        # Overridable checkpoint: per-frame masking measured 0% road with
        # ADE20k, but the CITYSCAPES checkpoint (dashcam road footage) is
        # used by the one-shot auto-ROI, where it only has to place a box.
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, use_safetensors=True)
        self.model.eval()

        # Look the class up; never hardcode it. Label ordering belongs to the
        # checkpoint, and a silently wrong index would mask the wrong pixels.
        id2label = self.model.config.id2label
        self.road_ids = [
            int(i) for i, lbl in id2label.items()
            if any(k in str(lbl).lower() for k in ROAD_LABEL_KEYWORDS)
        ]
        if not self.road_ids:
            raise RuntimeError(
                f"no road-like class in {model_name}. "
                f"Labels: {list(id2label.values())[:20]}")

    # ----------------------------------------------------------------------
    def mask(self, image: Image.Image) -> tuple[np.ndarray, float, list]:
        """Return (bool mask at original resolution, road fraction, top classes).

        The class histogram is the diagnostic that matters. ADE20k knows
        nothing about an F1 car filling the foreground of an onboard shot, so
        when coverage collapses the histogram tells you what it decided those
        pixels were instead - which is the difference between debugging and
        guessing.
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            logits = self.model(**inputs).logits

        # Logits come back at reduced resolution; upsample to the original.
        logits = torch.nn.functional.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        classes = logits.argmax(dim=1).squeeze(0).numpy()

        mask = np.isin(classes, self.road_ids)
        coverage = float(mask.mean())

        ids, counts = np.unique(classes, return_counts=True)
        order = np.argsort(-counts)[:6]
        id2label = self.model.config.id2label
        top = [
            {
                "label": str(id2label.get(int(ids[i]), ids[i])),
                "pct": round(float(counts[i] / classes.size) * 100, 1),
            }
            for i in order
        ]

        # Fallback: trust the crop.
        #
        # SegFormer returned 0% road on a Monaco onboard frame that was almost
        # entirely wet track. Rather than abstain on the images that matter
        # most, fall back to treating the whole ROI as surface - the operator
        # already told us where the track is by choosing the crop.
        #
        # Reported honestly via mask_source so a degraded reading is never
        # mistaken for a confident one.
        source = "segformer"
        if coverage < 0.02:
            mask = np.ones_like(mask, dtype=bool)
            coverage = 1.0
            source = "roi_fallback"

        return mask, coverage, top, source

    # ----------------------------------------------------------------------
    @staticmethod
    def apply(image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Black out everything that is not road.

        Black rather than transparent: CLIP has no alpha channel, and a
        uniform dark border is less likely to read as a scene feature than
        arbitrary leftover background.
        """
        arr = np.array(image)
        arr[~mask] = 0
        return Image.fromarray(arr)