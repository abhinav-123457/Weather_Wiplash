"""
suggestion.py - the message the brief actually asks for.

    "it gives a basic suggestion like 'Consider tire change soon.'"
    "a suggestion message (e.g. 'Track drying: tire change window
     approaching')"

Both examples are STATE + DIRECTION -> sentence. Nothing more.

This replaces a much larger strategy engine that also did compound selection,
two-compound rule tracking, tyre-age modelling and pit-loss economics. None
of that was asked for, every branch was another chance to produce a call that
could not be defended, and the estimates underneath it (lap-time gain per
compound, degradation) were the least grounded numbers in the project.

WHAT IS KEPT, AND WHY
  safety        slicks in standing water is a hazard, not an opinion. Cheap
                to state, impossible to argue with, and the system looks
                negligent without it.
  live weather  the one context input that can be VERIFIED: fetched from
                Open-Meteo for the circuit's coordinates and labelled with
                its source. Rain falling at the circuit is reported as a
                note - independent evidence, never fused into the label.

REMOVED 2026-08-13: the "slicks in ~N laps" crossover estimate. It rested
on BASE_DRYING_RATE, a tuning constant, and a horizon number nobody could
defend on stage is worse than no number. The drying physics still computes
and travels in the response as an off-screen diagnostic.

WHERE THE AI IS
  Not here. The intelligence is upstream: CLIP embeddings, a linear probe
  trained on 96 hand-labelled frames (83% cross-validated), and the temporal
  layer that turns per-frame states into a direction. Determining DRYING is
  the hard part. This module only says it in English.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..config import (BASE_DRYING_RATE, FULL_WET_THRESHOLD,
                      INTER_THRESHOLD, REFERENCE_VPD, SHADE_FACTOR,
                      SURFACE_DRAINAGE, TIRE_BAND, WIND_FACTOR_PER_MS)
from . import weather as met

_CIRCUITS: dict | None = None

# The three tyre families. Compounds (soft/medium/hard) are deliberately
# absent - choosing between them needs degradation modelling the brief does
# not ask for and this project cannot support honestly.
FAMILY_LABEL = {
    "SLICK": "slicks",
    "INTER": "intermediates",
    "FULL_WET": "full wets",
}


def circuits() -> dict:
    global _CIRCUITS
    if _CIRCUITS is None:
        for p in (Path(__file__).resolve().parents[3] / "circuits.json",
                  Path(__file__).resolve().parents[2] / "circuits.json"):
            if p.exists():
                _CIRCUITS = json.loads(p.read_text())
                break
        else:
            _CIRCUITS = {}
    return _CIRCUITS


# --------------------------------------------------------------------------
# Physics - real meteorology, kept because "window approaching" needs a horizon
# --------------------------------------------------------------------------
def saturation_vapour_pressure(t: float) -> float:
    """Magnus formula, hPa."""
    return 6.112 * math.exp(17.67 * t / (t + 243.5))


def drying_rate(circuit: dict, track_temp: float, air_temp: float,
                humidity: float, wind_speed: float) -> dict:
    """Wetness points removed per minute.

    Evaporation is driven by the vapour pressure deficit between the wet
    surface and the air, and accelerated by wind carrying saturated air away.

    Circuit contributes surface drainage and shade only. Climate is
    deliberately excluded - the caller supplies real temperature and
    humidity, so including it too would count the same effect twice.
    """
    vpd = max(saturation_vapour_pressure(track_temp)
              - saturation_vapour_pressure(air_temp) * (humidity / 100.0), 0.1)
    surface = SURFACE_DRAINAGE.get(circuit.get("surface_type", "permanent"), 1.0)
    shade = SHADE_FACTOR.get(circuit.get("shade", "none"), 1.0)
    wind = 1.0 + WIND_FACTOR_PER_MS * max(wind_speed, 0.0)

    return {
        "rate_per_min": round(
            BASE_DRYING_RATE * (vpd / REFERENCE_VPD) * wind * surface * shade, 2),
        "vpd_hpa": round(vpd, 1),
    }


def optimal_family(wetness: float) -> str:
    if wetness >= FULL_WET_THRESHOLD:
        return "FULL_WET"
    if wetness >= INTER_THRESHOLD:
        return "INTER"
    return "SLICK"


def family_of(tire: str) -> str:
    return "SLICK" if tire in ("SOFT", "MEDIUM", "HARD", "SLICK") else tire


def check_safety(current_tire: str, wetness: float) -> dict:
    """Is the fitted tyre usable in these conditions?

    Asked explicitly rather than inferred from whichever branches happened to
    exist. An earlier version had no branch for "slicks on a wet track", so
    it fell through to an economic rule and advised staying out to save a pit
    stop - on a car with no tread in standing water.
    """
    band = TIRE_BAND.get(current_tire)
    if not band:
        return {"status": "ok", "reason": None}

    if wetness > band["safe_max"]:
        what = ("Slicks have no tread"
                if family_of(current_tire) == "SLICK"
                else "Intermediates cannot clear this much water")
        return {"status": "unsafe",
                "reason": f"{what} — aquaplaning risk at wetness {wetness:.0f}."}

    if wetness < band["safe_min"]:
        what = ("Full wets" if current_tire == "FULL_WET" else "Intermediates")
        return {"status": "degrading",
                "reason": (f"{what} have no water left to cool them — "
                           f"they will overheat and go off.")}

    return {"status": "ok", "reason": None}


# --------------------------------------------------------------------------
# The suggestion
# --------------------------------------------------------------------------
# One line per (label, trend). Written out rather than assembled from
# fragments so every sentence a user can see is visible in one place and can
# be read aloud before it ever reaches a screen.
HEADLINES = {
    ("DRY", "STABLE"):     "Track dry: slicks optimal",
    ("DRY", "WETTING"):    "Track dry but wetness rising: watch closely",
    ("DRY", "DRYING"):     "Track dry: slicks optimal",

    ("DAMP", "STABLE"):    "Track damp: intermediates",
    ("DAMP", "DRYING"):    "Track drying: tire change window approaching",
    ("DAMP", "WETTING"):   "Conditions worsening: prepare for wets",

    ("WET", "STABLE"):     "Track wet: wet-weather tires",
    ("WET", "DRYING"):     "Track improving but still wet: hold",
    ("WET", "WETTING"):    "Conditions deteriorating: box for wets",

    ("DRYING", "DRYING"):  "Track drying: tire change window approaching",
    ("DRYING", "STABLE"):  "Track drying has stalled: hold current tires",
    ("DRYING", "WETTING"): "Drying reversed — wetness rising again",
}


def suggest(*, label: str, trend: str, wetness: float, confidence: float,
            circuit_key: str = "silverstone", current_tire: str = "INTER",
            track_temp: float | None = None, air_temp: float | None = None,
            humidity: float | None = None,
            wind_speed: float | None = None) -> dict:

    circuit = circuits().get(circuit_key, {})

    # Weather precedence, per field: operator-supplied > LIVE (Open-Meteo,
    # the circuit's actual current conditions) > per-circuit typical. The
    # source label travels with the values so nothing on screen is a guess
    # pretending to be a measurement.
    tw = circuit.get("typical_weather", {})
    supplied = [v is not None
                for v in (track_temp, air_temp, humidity, wind_speed)]
    lw = (met.live(circuit.get("lat"), circuit.get("lon"))
          if not all(supplied) else None)

    if air_temp is None:
        air_temp = lw["air_temp"] if lw else tw.get("air_temp", 18.0)
    if humidity is None:
        humidity = lw["humidity"] if lw else tw.get("humidity", 60.0)
    if wind_speed is None:
        wind_speed = lw["wind_speed"] if lw else tw.get("wind_speed", 2.0)
    if track_temp is None:
        # No public API measures track-surface temperature. Air temperature
        # is a conservative floor (asphalt runs warmer in any daylight) and
        # only feeds the off-screen drying diagnostic.
        track_temp = lw["air_temp"] if lw else tw.get("track_temp", 25.0)

    if all(supplied):
        source = "operator"
    elif any(supplied):
        source = "operator+live" if lw else "operator+typical"
    else:
        source = "live" if lw else "typical"

    dry = drying_rate(circuit, track_temp, air_temp, humidity, wind_speed)
    lap_time = float(circuit.get("lap_time_s", 90.0))

    current_family = family_of(current_tire)
    want = optimal_family(wetness)
    safety = check_safety(current_tire, wetness)

    headline = HEADLINES.get((label, trend))
    if headline is None:
        headline = f"Track {label.lower()}"

    detail = (f"on {FAMILY_LABEL[current_family]}, "
              f"{FAMILY_LABEL[want]} suited to current conditions"
              if want != current_family else None)

    # Urgency: safety first, then a pending change, then trend, then quiet.
    if safety["status"] == "unsafe":
        urgency = "URGENT"
        headline = f"Unsafe tires: box for {FAMILY_LABEL[want]}"
        detail = safety["reason"]
    elif safety["status"] == "degrading":
        urgency = "URGENT"
        detail = safety["reason"]
    elif want != current_family:
        urgency = "ADVISORY"
    elif trend in ("DRYING", "WETTING"):
        urgency = "ADVISORY"
    else:
        urgency = "INFO"

    notes = []
    if confidence < 0.45:
        notes.append("Low classifier confidence — treat as provisional.")
    # Rain at the circuit is INDEPENDENT evidence: it comes from a weather
    # station, not from the pixels. Reported as a note, never fused into
    # the label - same rule as scene context and the road probe.
    if lw and lw.get("rain_mm", 0) > 0:
        notes.append(f"Rain falling at circuit now: "
                     f"{lw['rain_mm']:.1f} mm/h (live).")

    return {
        "headline": headline,
        "detail": detail,
        "urgency": urgency,
        "notes": notes,
        "current_tire": current_family,
        "suggested_tire": want,
        "change_needed": want != current_family,
        "safety": safety["status"],
        # Kept as a key for API compatibility; the crossover estimate was
        # removed from the product (see module docstring).
        "laps_to_crossover": None,
        # What the suggestion was derived from, so nothing on screen is
        # unexplainable.
        "basis": (f"{label} · wetness {wetness:.0f} · {trend.lower()}"),
        "_wetness": round(wetness, 1),
        "drying": dry,
        "weather": {
            "track_temp": round(track_temp, 1), "air_temp": round(air_temp, 1),
            "humidity": round(humidity, 0), "wind_speed": round(wind_speed, 1),
            "rain_mm": (lw or {}).get("rain_mm"),
            "source": source,
        },
        "circuit": {"key": circuit_key,
                    "name": circuit.get("name", circuit_key),
                    "lap_time_s": lap_time},
    }