#!/usr/bin/env python3
"""
test_trend.py - does the trend graph actually say the right thing?

The classifier has a number you can quote: 83% cross-validated on 96
labelled frames. The TREND had no such number - it was only ever eyeballed
on real footage, where the ground truth is a matter of opinion.

This fixes that. Each case is a sequence with a KNOWN shape, so the correct
trend and label path are not debatable, and each states its expectation in
plain English. You do not need to know F1 to read the result.

    python tools\\test_trend.py

No backend needed - this drives the temporal layer directly.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("cfg", ROOT / "backend/app/config.py")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
pkg = types.ModuleType("app")
pkg.__path__ = []
sys.modules["app"] = pkg
sys.modules["app.config"] = cfg
src = (ROOT / "backend/app/engine/temporal.py").read_text().replace(
    "from ..config import", "from app.config import")
temporal = types.ModuleType("temporal")
exec(compile(src, "temporal", "exec"), temporal.__dict__)


def state_for(w: float) -> str:
    """Stand in for the classifier, so the TREND is what is under test."""
    if w < 45:
        return "DRY"
    if w < 65:
        return "DAMP"
    return "WET"


def run_sequence(values, conf=0.85):
    s = temporal.SessionState("t")
    rows = []
    for w in values:
        rows.append(s.add(w, state_for(w), conf))
    path = []
    for r in rows:
        if not path or path[-1] != r["label"]:
            path.append(r["label"])
    return rows, path


def ramp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def jitter(base, amp, n, seed=1):
    # Deterministic pseudo-noise: a test that changes answer between runs is
    # not a test.
    out, x = [], seed
    for _ in range(n):
        x = (1103515245 * x + 12345) % 2147483648
        out.append(base + amp * ((x / 2147483648) * 2 - 1))
    return out


CASES = [
    {
        "name": "Steady downpour - nothing changing",
        "why": "A stable track must not produce a trend. Reporting DRYING "
               "here would put a tyre call on screen for a decision that "
               "does not exist.",
        "expect": "trend STABLE throughout, label stays WET",
        "seq": jitter(85, 6, 14),
        "check": lambda rows, path: (path == ["WET"]
                                     and all(r["trend"] == "STABLE" for r in rows[5:])),
    },
    {
        "name": "Genuine drying - soaked to dry over 16 frames",
        "why": "The whole point of the system. It must detect the direction "
               "and pass through DRYING on the way.",
        "expect": "DRYING appears, and the path ends at DRY",
        "seq": ramp(90, 15, 16),
        "check": lambda rows, path: ("DRYING" in path and path[-1] == "DRY"
                                     and any(r["trend"] == "DRYING" for r in rows)),
    },
    {
        "name": "Rain arriving - dry to soaked over 12 frames",
        "why": "The safety-critical direction. A symmetric-hysteresis bug "
               "once left the label on DRY through an entire simulated "
               "downpour.",
        "expect": "WETTING detected, and the path ends at WET",
        "seq": ramp(10, 90, 12),
        "check": lambda rows, path: (path[-1] == "WET"
                                     and any(r["trend"] == "WETTING" for r in rows)),
    },
    {
        "name": "Noisy damp track, no real change",
        "why": "Real classifier output is noisy. If noise flips the label, "
               "the banner flickers and the tool becomes unusable.",
        "expect": "label never changes",
        "seq": jitter(55, 9, 14, seed=7),
        "check": lambda rows, path: len(path) == 1,
    },
    {
        "name": "Drying that stalls halfway",
        "why": "Weather does not always cooperate. Once the drop stops, the "
               "system must stop promising a slick window.",
        "expect": "DRYING at first, then back to STABLE by the end",
        "seq": ramp(90, 60, 8) + jitter(60, 3, 8, seed=3),
        "check": lambda rows, path: (any(r["trend"] == "DRYING" for r in rows[:8])
                                     and rows[-1]["trend"] == "STABLE"),
    },
    {
        "name": "Single spurious spike in a stable sequence",
        "why": "One bad frame must not move the call. This is what "
               "hysteresis exists for.",
        "expect": "label unchanged despite the spike",
        "seq": [30] * 6 + [88] + [30] * 6,
        "check": lambda rows, path: len(path) == 1,
    },
    {
        "name": "First frames report no trend",
        "why": "A slope needs a full window. Fitting a partial one measured "
               "the smoothing settling down, not the track, and invented a "
               "downward trend at the start of every session.",
        "expect": "STABLE until the window fills",
        "seq": ramp(90, 20, 12),
        "check": lambda rows, path: all(
            r["trend"] == "STABLE" for r in rows[:cfg.SLOPE_WINDOW - 1]),
    },
]


def main() -> int:
    passed = failed = 0
    print("=" * 74)
    print("TREND LAYER - known-shape sequences")
    print("=" * 74)

    for c in CASES:
        rows, path = run_sequence(c["seq"])
        ok = bool(c["check"](rows, path))
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)

        print(f"\n{c['name']}")
        print(f"  why it matters : {c['why']}")
        print(f"  expected       : {c['expect']}")
        print(f"  wetness in     : {rows[0]['wetness_raw']:.0f} -> "
              f"{rows[-1]['wetness_raw']:.0f}  ({len(rows)} frames)")
        print(f"  label path     : {' -> '.join(path)}")
        trends = []
        for r in rows:
            if not trends or trends[-1] != r["trend"]:
                trends.append(r["trend"])
        print(f"  trend path     : {' -> '.join(trends)}")
        print(f"  RESULT         : {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 74)
    print(f"{passed} passed, {failed} failed   "
          f"({passed}/{passed + failed} = {passed / (passed + failed) * 100:.0f}%)")
    if not failed:
        print("\nThe trend layer now has a number you can quote, the same way")
        print("the classifier has 83% cross-validated accuracy.")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
