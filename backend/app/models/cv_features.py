"""
cv_features.py - physical wetness signals from classical computer vision.

These earn their place only if they fail DIFFERENTLY from CLIP. An earlier
version weighted absolute darkness at 0.50 and scored a bone-dry photo of
fresh dark tarmac at 84.5/100 - while the CLIP prompt for "damp" also
mentioned "dark surface". Two signals, one shared mistake, no independence.

So: no absolute brightness anywhere in the score. The two physical
properties that actually separate wet asphalt from dry asphalt of any colour:

  1. Reflectivity - a water film is specular, dry aggregate is diffuse
  2. Texture     - water fills surface pores and smooths micro-relief

Brightness is still reported, purely as a diagnostic. It does not vote.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import (CV_SPECULAR_WEIGHT, CV_SPRAY_WEIGHT, CV_TEXTURE_WEIGHT,
                      TEXTURE_VAR_DRY, TEXTURE_VAR_WET)


def specular_score(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Standing water is mirror-like and throws bright highlights.

    Returns (score 0-100, raw bright-pixel ratio).
    """
    if not mask.any():
        return 0.0, 0.0
    hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS)
    bright = (hls[:, :, 1] > 200) & mask
    ratio = float(bright.sum() / max(mask.sum(), 1))
    # Even soaked asphalt rarely exceeds ~15% blown-out pixels, so scale
    # against that rather than against 100%.
    return float(np.clip(ratio / 0.15 * 100.0, 0.0, 100.0)), ratio


def texture_score(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Wet asphalt loses micro-texture as water fills the surface pores.

    Laplacian variance measures high-frequency detail. Dry aggregate is
    rough and scores high; a water film smooths it and scores low. Crucially
    this is INDEPENDENT of how dark the asphalt is, which is exactly what the
    brightness heuristic got wrong.

    Returns (wetness score 0-100, raw Laplacian variance).
    """
    if not mask.any():
        return 0.0, 0.0

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    var = float(lap[mask].var())

    # Log scale: texture variance spans orders of magnitude, so a linear
    # mapping would put almost every real reading at one end.
    lo, hi = np.log10(TEXTURE_VAR_WET), np.log10(TEXTURE_VAR_DRY)
    pos = (np.log10(max(var, 1.0)) - lo) / (hi - lo)      # 0 = wet, 1 = dry
    return float(np.clip((1.0 - pos) * 100.0, 0.0, 100.0)), var


def spray_score(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Spray plumes - the signal race control actually watches.

    Looked for ABOVE the surface, off-road: spray is airborne. High
    brightness with low saturation is the signature of water mist.
    """
    h = rgb.shape[0]
    above = np.zeros(mask.shape, dtype=bool)
    above[: h // 2, :] = True
    region = above & ~mask
    if region.sum() < 100:
        return 0.0, 0.0

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    spray = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 180) & region
    ratio = float(spray.sum() / max(region.sum(), 1))
    return float(np.clip(ratio / 0.25 * 100.0, 0.0, 100.0)), ratio


def brightness_diagnostic(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Mean grey of the road surface. DIAGNOSTIC ONLY - never scored.

    Useful for spotting why a frame behaved oddly, and for calibration
    later. Deliberately excluded from the wetness estimate.
    """
    if not mask.any():
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return round(float(gray[mask].mean()), 1)


def extract(rgb: np.ndarray, mask: np.ndarray,
            camera_type: str = "trackside") -> dict:
    spec, spec_raw = specular_score(rgb, mask)
    tex, tex_raw = texture_score(rgb, mask)

    # Onboard cameras look down the track - there is no meaningful region
    # "above the surface" to find plumes in, so running it would add noise.
    # CV_TEXTURE_WEIGHT is 0.0 - texture tracked camera sharpness, not
    # wetness (see config). Left in the arithmetic so re-enabling it is a
    # one-line change if blur normalisation is ever added.
    if camera_type == "trackside":
        spray, spray_raw = spray_score(rgb, mask)
        total_w = CV_SPECULAR_WEIGHT + CV_TEXTURE_WEIGHT + CV_SPRAY_WEIGHT
        combined = (CV_SPECULAR_WEIGHT * spec
                    + CV_TEXTURE_WEIGHT * tex
                    + CV_SPRAY_WEIGHT * spray) / total_w
    else:
        # Onboard has no usable spray region, so specular carries CV alone.
        spray, spray_raw = 0.0, 0.0
        total_w = CV_SPECULAR_WEIGHT + CV_TEXTURE_WEIGHT
        combined = (CV_SPECULAR_WEIGHT * spec
                    + CV_TEXTURE_WEIGHT * tex) / total_w

    return {
        "specular": round(spec, 1),
        "texture": round(tex, 1),
        "spray": round(spray, 1),
        "combined": round(float(np.clip(combined, 0.0, 100.0)), 1),
        # Raw values for calibration against RoadSaW. The normalised scores
        # above depend on constants that are still guesses; these do not.
        "raw": {
            "specular_ratio": round(spec_raw, 5),
            "laplacian_var": round(tex_raw, 1),
            "spray_ratio": round(spray_raw, 5),
            "mean_brightness": brightness_diagnostic(rgb, mask),
        },
    }
