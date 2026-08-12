"""
temporal.py - the layer that turns per-frame states into a direction.

THE PREMISE OF THE WHOLE SYSTEM
-------------------------------
The brief asks for four labels: Dry, Damp, Wet, Drying. Three of those
describe how much water is on the track. One describes which way it is
moving - and a single frozen frame cannot answer that. Identical pixels
mean opposite things depending on what came before.

Measured evidence, from 96 labelled frames scored by the probe:

    dry   94%      wet   94%      damp   61%

Damp errs in both directions (4 called dry, 8 called wet) because it is not
a distinct visual state - it is the region between two states. This layer is
what resolves it:

    wetness 45, arrived from 80  ->  DRYING   slick window opening
    wetness 45, arrived from 15  ->  WETTING  get inters on

Same frame. Same score. Opposite tire call.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import (CONFIDENCE_MIN, DRYING_MIN_WETNESS, EMA_ALPHA,
                      HYSTERESIS_FRAMES, LOW_CONFIDENCE_ALPHA_SCALE,
                      HYSTERESIS_FRAMES_WORSENING, LABEL_SEVERITY,
                      SLOPE_DRYING_THRESHOLD, SLOPE_WETTING_THRESHOLD,
                      SLOPE_WINDOW)


def _slope(values: list[float], window: int) -> float:
    """Least-squares slope over the last `window` points, per frame.

    Fitted rather than taking (last - first): endpoint differencing lets a
    single noisy frame at either end dictate the whole trend, which is
    exactly the flicker this layer exists to remove.
    """
    # Require the FULL window, not a partial one.
    #
    # Frame 1's smoothed value equals its raw value (nothing to average
    # against yet), so an early partial fit is measuring the EMA settling
    # rather than the track changing. That transient produced a phantom
    # downward slope in the first few frames of every session.
    if len(values) < window:
        return 0.0
    n = window
    y = np.asarray(values[-n:], dtype=float)
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = float(((x - xm) ** 2).sum())
    if denom == 0.0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


class SessionState:
    """Frame history for one analysis session.

    Sessions must be explicit. Without one, a second demo run would append
    to the first and the trend line would be nonsense.
    """

    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.created = time.time()
        self.touched = time.time()

        self.raw: list[float] = []
        self.smooth: list[float] = []
        self.states: list[str] = []      # per-frame classifier output
        self.labels: list[str] = []      # committed, after hysteresis

        self._pending: str | None = None
        self._pending_n = 0
        self._pending_worsening = False

        # Kept so a tyre call can be recomputed from the current condition
        # without re-uploading a frame.
        self.last_trend = "STABLE"
        self.last_confidence = 1.0

        # Dry-reference calibration. Owned by the session because the offset
        # is a property of this camera view, not of the model. last_uncalibrated
        # exists so setting the reference twice does not compound: the offset
        # must always be computed from what the classifier actually said, not
        # from an already-adjusted number.
        self.reference_offset: float | None = None
        self.last_uncalibrated: float | None = None

        # Auto-ROI follow: the detector's current box, and the previous
        # frame's full-scene embedding (a cut = a big embedding jump, which
        # is the trigger to re-detect). Session-owned because the box is a
        # property of the footage being watched, not of the model.
        self.auto_roi_box: list | None = None
        self.last_scene_emb = None

        # Run lengths, for the Why panel. "Sustained for 5 frames" is the
        # difference between a trend and a coincidence, and it is the one
        # piece of evidence a viewer can check against the chart by eye.
        self._trend_run = 0
        self._label_run = 0

    # ------------------------------------------------------------------
    def add(self, wetness: float, state: str, confidence: float) -> dict:
        self.touched = time.time()

        # A frame the classifier could not call is weak evidence, not no
        # evidence: it still nudges the score, at reduced weight, but it is
        # barred below from moving the label at all.
        low_conf = confidence < CONFIDENCE_MIN
        alpha = EMA_ALPHA * (LOW_CONFIDENCE_ALPHA_SCALE if low_conf else 1.0)

        # Exponential moving average. Raw per-frame scores are noisy enough
        # that an unsmoothed slope is dominated by frame-to-frame jitter.
        prev = self.smooth[-1] if self.smooth else wetness
        smooth = alpha * wetness + (1.0 - alpha) * prev

        self.raw.append(float(wetness))
        self.smooth.append(float(smooth))
        self.states.append(state)

        slope = _slope(self.smooth, SLOPE_WINDOW)

        # ---- trend ----
        # Guarded by DRYING_MIN_WETNESS: a small negative slope on an
        # already-dry track is noise, not drying. There is nothing left to
        # dry, and calling it DRYING would put a tire recommendation on the
        # screen for a decision nobody needs to make.
        if slope <= SLOPE_DRYING_THRESHOLD and smooth > DRYING_MIN_WETNESS:
            trend = "DRYING"
        elif slope >= SLOPE_WETTING_THRESHOLD:
            trend = "WETTING"
        else:
            trend = "STABLE"

        # ---- the four required labels ----
        # State supplies three; the fourth comes from direction. DRYING
        # outranks DAMP and WET because it is the actionable one - it is
        # what opens the crossover window.
        if state == "DRY":
            candidate = "DRY"
        elif trend == "DRYING":
            candidate = "DRYING"
        else:
            candidate = state

        # An uncertain frame holds the current label rather than changing it.
        # Note this comes AFTER the score update, so a genuine shift still
        # shows in the chart and the slope - it simply has to be confirmed by
        # a frame the classifier can actually call before the banner moves.
        label_held = False
        if low_conf and self.labels:
            committed = self.labels[-1]
            self.labels.append(committed)
            # Only a SUPPRESSION worth reporting if the uncertain frame
            # actually wanted a different label. An uncertain frame that
            # agrees with the current call changed nothing, and warning about
            # it makes a correct answer look like a problem.
            label_held = candidate != committed
        else:
            committed = self._commit(candidate)

        self._trend_run = self._trend_run + 1 if trend == self.last_trend else 1
        self._label_run = (self._label_run + 1
                           if len(self.labels) > 1 and committed == self.labels[-2]
                           else 1)
        self.last_trend = trend
        self.last_confidence = float(confidence)

        return {
            "label": committed,
            "state": state,
            "trend": trend,
            "low_confidence": bool(low_conf),
            "label_held": bool(label_held),
            "wetness": round(smooth, 1),
            "wetness_raw": round(float(wetness), 1),
            "slope": round(slope, 2),
            "confidence": round(float(confidence), 3),
            "frame": len(self.raw),
            "pending_label": self._pending,
            "trend_frames": self._trend_run,
            "label_frames": self._label_run,
            # Evidence is built HERE, from the same values the decision used,
            # so the explanation cannot drift from the behaviour. A panel that
            # re-derives its own reasoning in the frontend will eventually
            # describe a decision the backend did not make.
            "evidence": self._evidence(slope, smooth, confidence, trend,
                                       low_conf, label_held),
            "chart_data": [
                {"frame": i + 1, "raw": round(r, 1), "smooth": round(s, 1)}
                for i, (r, s) in enumerate(zip(self.raw, self.smooth))
            ],
        }

    # ------------------------------------------------------------------
    def _evidence(self, slope: float, smooth: float, confidence: float,
                  trend: str, low_conf: bool, label_held: bool) -> list[dict]:
        """The checks that produced this trend, pass or fail.

        Failing checks are included deliberately. Knowing why the system is
        NOT calling a trend is as useful as knowing why it is - "slope only
        -1.2, threshold is -2.5" tells an operator the track is moving but
        not yet enough to act on.
        """
        ev = []

        if len(self.smooth) < SLOPE_WINDOW:
            ev.append({
                "check": "enough frames for a trend",
                "detail": f"{len(self.smooth)} of {SLOPE_WINDOW} needed",
                "pass": False,
            })
            return ev

        ev.append({
            "check": "falling fast enough to call drying",
            "detail": f"slope {slope:+.1f} vs threshold "
                      f"{SLOPE_DRYING_THRESHOLD:+.1f} per frame",
            "pass": slope <= SLOPE_DRYING_THRESHOLD,
        })
        ev.append({
            "check": "wet enough for drying to matter",
            "detail": f"wetness {smooth:.1f} vs floor {DRYING_MIN_WETNESS:.0f}",
            "pass": smooth > DRYING_MIN_WETNESS,
        })
        ev.append({
            "check": "rising fast enough to call wetting",
            "detail": f"slope {slope:+.1f} vs threshold "
                      f"{SLOPE_WETTING_THRESHOLD:+.1f} per frame",
            "pass": slope >= SLOPE_WETTING_THRESHOLD,
        })
        ev.append({
            "check": "classifier confident enough to count",
            "detail": f"{confidence:.2f} vs minimum {CONFIDENCE_MIN:.2f}",
            "pass": not low_conf,
        })
        ev.append({
            "check": f"trend held",
            "detail": f"{self._trend_run} consecutive frame"
                      f"{'s' if self._trend_run != 1 else ''} as {trend}",
            "pass": self._trend_run >= 2,
        })
        if label_held:
            ev.append({
                "check": "label change suppressed",
                "detail": "uncertain frame wanted a different call - held",
                "pass": False,
            })
        return ev

    # ------------------------------------------------------------------
    def _commit(self, candidate: str) -> str:
        """Adopt a new label once it has earned it - asymmetrically.

        Without any delay the banner flickers between adjacent states on
        noisy frames, which looks broken and is wrong: a tire call that
        changes every frame is not a call.

        But the delay must not be symmetric. Too-early slicks on a damp
        track is a spin; too-late is a few seconds a lap. So:

            worsening  -> commit almost immediately
            improving  -> require sustained evidence

        Direction is judged by LABEL_SEVERITY, not by string equality. An
        earlier version counted consecutive IDENTICAL candidates, so a track
        going DRY -> DAMP -> WET reset the counter at every step and never
        committed - the label sat on DRY through a whole downpour.
        """
        if not self.labels:
            self.labels.append(candidate)
            return candidate

        current = self.labels[-1]

        if candidate == current:
            self._pending = None
            self._pending_n = 0
            self.labels.append(current)
            return current

        worsening = (LABEL_SEVERITY.get(candidate, 0)
                     > LABEL_SEVERITY.get(current, 0))
        needed = (HYSTERESIS_FRAMES_WORSENING if worsening
                  else HYSTERESIS_FRAMES)

        # Count evidence for the DIRECTION, not for one exact label, so a
        # track escalating through several states still accumulates.
        if self._pending is not None and worsening == self._pending_worsening:
            self._pending_n += 1
        else:
            self._pending_n = 1
        self._pending = candidate
        self._pending_worsening = worsening

        if self._pending_n >= needed:
            self._pending = None
            self._pending_n = 0
            self.labels.append(candidate)
            return candidate

        self.labels.append(current)
        return current

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.raw.clear()
        self.smooth.clear()
        self.states.clear()
        self.labels.clear()
        self._pending = None
        self._pending_n = 0
        self._pending_worsening = False
        # A new session usually means new footage, and the offset belongs to
        # the old camera view. Re-marking a dry frame takes one click; a stale
        # offset silently corrupting every reading is much worse.
        self.reference_offset = None
        self.last_uncalibrated = None
        self.auto_roi_box = None
        self.last_scene_emb = None
        self.last_trend = "STABLE"
        self.last_confidence = 1.0
        self._trend_run = 0
        self._label_run = 0
        self.created = time.time()


class SessionStore:
    """In-memory session registry with age-based eviction."""

    def __init__(self, max_age_s: float = 3600.0) -> None:
        self._s: dict[str, SessionState] = {}
        self.max_age_s = max_age_s

    def get(self, session_id: str) -> SessionState:
        self._evict()
        if session_id not in self._s:
            self._s[session_id] = SessionState(session_id)
        return self._s[session_id]

    def reset(self, session_id: str) -> None:
        if session_id in self._s:
            self._s[session_id].reset()

    def _evict(self) -> None:
        now = time.time()
        for k in [k for k, v in self._s.items()
                  if now - v.touched > self.max_age_s]:
            del self._s[k]