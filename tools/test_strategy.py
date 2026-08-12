#!/usr/bin/env python3
"""
test_strategy.py - self-checking tests for the strategy engine.

You should not need to know F1 strategy to tell whether this works, so each
case states in plain English what a sensible answer looks like and the script
checks it. Green means the engine agrees with common sense.

    python tools\\test_strategy.py

No backend needed - this calls the engine directly.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load config + strategy without dragging in torch/transformers.
spec = importlib.util.spec_from_file_location("cfg", ROOT / "backend/app/config.py")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
pkg = types.ModuleType("app")
pkg.__path__ = []
sys.modules["app"] = pkg
sys.modules["app.config"] = cfg
src = (ROOT / "backend/app/engine/strategy.py").read_text().replace(
    "from ..config import", "from app.config import")
strategy = types.ModuleType("strategy")
strategy.__file__ = str(ROOT / "backend/app/engine/strategy.py")
exec(compile(src, "strategy", "exec"), strategy.__dict__)


# --------------------------------------------------------------------------
# Each case: the situation, why the answer matters, and what "sensible" means.
# --------------------------------------------------------------------------
CASES = [
    {
        "name": "Torrential rain while on slick tyres",
        "why": "Slicks have no tread. On standing water the car aquaplanes. "
               "This is the most dangerous situation in the whole system.",
        "expect": "urgent warning, and a wet-weather tyre suggested",
        "inputs": dict(label="WET", trend="WETTING", wetness=92, confidence=.9,
                       circuit_key="spa", current_tire="MEDIUM", tire_age=8,
                       current_lap=10, total_laps=44, track_temp=16,
                       air_temp=14, humidity=96, wind_speed=2),
        "check": lambda r: (r["urgency"] == "URGENT"
                            and r["suggested_tire"] in ("INTER", "FULL_WET")),
    },
    {
        "name": "Track fully dry but still on intermediates, 20 laps left",
        "why": "Intermediates are cooled by the water they clear. On a dry "
               "track they overheat and fall apart within a few laps.",
        "expect": "a slick suggested, a change flagged, and a warning about "
                  "the intermediates",
        "inputs": dict(label="DRY", trend="DRYING", wetness=25, confidence=.88,
                       circuit_key="silverstone", current_tire="INTER",
                       tire_age=14, current_lap=32, total_laps=52,
                       track_temp=33, air_temp=20, humidity=45, wind_speed=5),
        "check": lambda r: (r["suggested_tire"] in ("SOFT", "MEDIUM", "HARD")
                            and r["change_needed"]
                            and any("overheat" in n.lower() for n in r["notes"])),
    },
    {
        "name": "Track drying with plenty of race left",
        "why": "The crossover is the decisive call of any wet-to-dry race. "
               "The engine should say how many laps until slicks are right.",
        "expect": "a lap count for the slick window",
        "inputs": dict(label="DRYING", trend="DRYING", wetness=60,
                       confidence=.82, circuit_key="silverstone",
                       current_tire="INTER", tire_age=7, current_lap=25,
                       total_laps=52, track_temp=30, air_temp=19,
                       humidity=55, wind_speed=4),
        "check": lambda r: (r["laps_to_crossover"] is not None
                            and r["laps_to_crossover"] > 0),
    },
    {
        "name": "Same drying track, but only 3 laps remain",
        "why": "A pit stop costs about 20 seconds. Near the end of a race "
               "there is not enough track left to win that back, so the "
               "right answer flips even though conditions are identical.",
        "expect": "told to stay out, and NOT urgent",
        "inputs": dict(label="DRYING", trend="DRYING", wetness=60,
                       confidence=.82, circuit_key="silverstone",
                       current_tire="INTER", tire_age=7, current_lap=49,
                       total_laps=52, track_temp=30, air_temp=19,
                       humidity=55, wind_speed=4),
        "check": lambda r: ("stay out" in r["message"].lower()
                            and r["urgency"] != "URGENT"),
    },
    {
        "name": "Dry track, already on the right tyre",
        "why": "The system must stay quiet when nothing needs doing. A tool "
               "that always shouts gets ignored.",
        "expect": "no urgency",
        "inputs": dict(label="DRY", trend="STABLE", wetness=10, confidence=.93,
                       circuit_key="suzuka", current_tire="MEDIUM",
                       tire_age=6, current_lap=12, total_laps=53,
                       track_temp=28, air_temp=22, humidity=50, wind_speed=3),
        "check": lambda r: r["urgency"] == "INFO",
    },
    {
        "name": "Monaco in a downpour - can it dry before the flag?",
        "why": "Monaco drains badly, sits in shade, and the race is long. "
               "Quoting a crossover 78 laps away would be arithmetically "
               "right and operationally useless.",
        "expect": "no crossover lap quoted",
        "inputs": dict(label="WET", trend="STABLE", wetness=90, confidence=.9,
                       circuit_key="monaco", current_tire="FULL_WET",
                       tire_age=5, current_lap=30, total_laps=78,
                       track_temp=17, air_temp=15, humidity=95, wind_speed=1),
        "check": lambda r: r["laps_to_crossover"] is None,
    },
    {
        "name": "Hot dry circuit dries faster than a cold damp one",
        "why": "Evaporation is driven by temperature and humidity. Bahrain "
               "in the sun must dry faster than Monaco in the cold.",
        "expect": "Bahrain's drying rate higher than Monaco's",
        "inputs": None,      # handled specially below
        "check": None,
    },
]


def message_agrees_with_tyre(r: dict) -> bool:
    """The wording must not contradict the recommendation.

    Added after a real miss: on a flooded track the engine suggested
    FULL_WET while the message read "Box for intermediates". Both the
    urgency check and the tyre check passed, because neither compared the
    two against each other.
    """
    import re
    msg = r["message"].lower()
    tyre = r["suggested_tire"]
    slicks = ("SOFT", "MEDIUM", "HARD")

    # Only inspect the tyre the message tells you to FIT. A first attempt
    # matched any mention of a tyre and failed a correct message - "Wetness
    # rising on slicks. Box for full wets." names the current tyre first,
    # which is not what is being recommended.
    m = re.search(r"box for ([a-z ]+?)\s*[.,]", msg)
    if m:
        named = m.group(1).strip()
        if named == "intermediates" and tyre != "INTER":
            return False
        if named == "full wets" and tyre != "FULL_WET":
            return False

    if "slicks optimal" in msg and tyre not in slicks:
        return False
    if "stay on intermediates" in msg and tyre != "INTER":
        return False
    return True


def run() -> int:
    passed = failed = 0
    print("=" * 72)
    print("STRATEGY ENGINE - plain-English checks")
    print("=" * 72)

    for case in CASES:
        print(f"\n{case['name']}")
        print(f"  why it matters : {case['why']}")
        print(f"  sensible answer: {case['expect']}")

        if case["inputs"] is None:
            # The comparative physics check.
            hot = strategy.drying_rate(
                strategy.circuits().get("bahrain", {}), 40, 30, 25, 4)
            cold = strategy.drying_rate(
                strategy.circuits().get("monaco", {}), 16, 14, 92, 1)
            ok = hot["rate_per_min"] > cold["rate_per_min"]
            print(f"  engine said    : Bahrain {hot['rate_per_min']}/min vs "
                  f"Monaco {cold['rate_per_min']}/min")
        else:
            r = strategy.recommend(**case["inputs"])
            ok = bool(case["check"](r))
            if ok and not message_agrees_with_tyre(r):
                ok = False
                print("  ! message contradicts the suggested tyre")
            print(f"  engine said    : [{r['urgency']}] {r['message']}")
            for n in r["notes"]:
                print(f"                   - {n}")
            print(f"                   tyre {r['current_tire']} -> "
                  f"{r['suggested_tire']}, crossover="
                  f"{r['laps_to_crossover']}")

        print(f"  RESULT         : {'PASS' if ok else 'FAIL'}")
        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)

    print("\n" + "=" * 72)
    print(f"{passed} passed, {failed} failed")
    if failed:
        print("\nA failure means the engine disagreed with the plain-English")
        print("expectation above it. Read that line - it says what was wanted.")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
