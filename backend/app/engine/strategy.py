"""
strategy.py - turn a track condition into a tire call.

The vision stack answers "how wet, and which way is it moving". This answers
"so what do I put on the car". It is the layer the problem statement actually
names: a suggestion message, not just a label.

Everything here is manual input plus physics. None of it is inferable from a
photograph - a camera cannot see your tire age, your lap count or the
humidity. That is not a gap: a real pit wall already has all of it on screen,
and this takes the same inputs a race engineer already has.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..config import (BASE_DRYING_RATE, COMPOUND_LAPS_MEDIUM,
                      COMPOUND_LAPS_SOFT, DRY_THRESHOLD, FULL_WET_THRESHOLD,
                      INTER_THRESHOLD, REFERENCE_VPD, SEVERITY_FACTOR,
                      SHADE_FACTOR, SURFACE_DRAINAGE, TIRE_BAND,
                      TIME_GAIN_SLICK_VS_INTER, WIND_FACTOR_PER_MS)

_CIRCUITS: dict | None = None


def circuits() -> dict:
    global _CIRCUITS
    if _CIRCUITS is None:
        path = Path(__file__).resolve().parents[3] / "circuits.json"
        if not path.exists():
            path = Path(__file__).resolve().parents[2] / "circuits.json"
        _CIRCUITS = json.loads(path.read_text()) if path.exists() else {}
    return _CIRCUITS


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
def saturation_vapour_pressure(t_celsius: float) -> float:
    """Magnus formula, hPa. Real meteorology, not a fudge factor."""
    return 6.112 * math.exp(17.67 * t_celsius / (t_celsius + 243.5))


def drying_rate(circuit: dict, track_temp: float, air_temp: float,
                humidity: float, wind_speed: float) -> dict:
    """Wetness points removed per minute.

    Water leaves the track by evaporation, driven by the vapour pressure
    deficit between the wet surface and the air, and accelerated by wind
    carrying saturated air away.

    Circuit contributes only SURFACE and SHADE here - climate is deliberately
    excluded, because the caller is already supplying the actual temperature
    and humidity. Including both would count the same effect twice.
    """
    es_track = saturation_vapour_pressure(track_temp)
    es_air = saturation_vapour_pressure(air_temp)
    vpd = max(es_track - es_air * (humidity / 100.0), 0.1)

    surface = SURFACE_DRAINAGE.get(circuit.get("surface_type", "permanent"), 1.0)
    shade = SHADE_FACTOR.get(circuit.get("shade", "none"), 1.0)
    wind = 1.0 + WIND_FACTOR_PER_MS * max(wind_speed, 0.0)

    rate = BASE_DRYING_RATE * (vpd / REFERENCE_VPD) * wind * surface * shade

    return {
        "rate_per_min": round(rate, 2),
        "vpd_hpa": round(vpd, 1),
        "factors": {
            "surface": surface, "shade": shade, "wind": round(wind, 2),
        },
    }


def laps_to_target(wetness: float, target: float, rate_per_min: float,
                   lap_time_s: float) -> int | None:
    """How many laps until the track reaches `target` wetness."""
    if wetness <= target:
        return 0
    if rate_per_min <= 0.01:
        return None                      # not drying at all
    minutes = (wetness - target) / rate_per_min
    return max(1, math.ceil(minutes * 60.0 / lap_time_s))


# --------------------------------------------------------------------------
# Tire selection
# --------------------------------------------------------------------------
def optimal_family(wetness: float) -> str:
    """Which tire FAMILY the current surface calls for."""
    if wetness >= FULL_WET_THRESHOLD:
        return "FULL_WET"
    if wetness >= INTER_THRESHOLD:
        return "INTER"
    return "SLICK"


def assess_current_tire(current_tire: str, wetness: float) -> dict:
    """Is the tyre on the car actually usable in these conditions?

    Asked explicitly rather than inferred from whichever branches happened to
    be written. The previous version had no branch for "slicks on a wet
    track", so it fell through to the economic rule and advised staying out
    to save a pit stop - on a car with no tread in standing water.
    """
    band = TIRE_BAND.get(current_tire)
    if not band:
        return {"status": "unknown", "reason": None}

    if wetness > band["safe_max"]:
        if current_tire in ("SOFT", "MEDIUM", "HARD"):
            return {
                "status": "unsafe",
                "reason": (f"Slicks have no tread. At wetness {wetness:.0f} "
                           f"there is standing water — aquaplaning risk."),
            }
        return {
            "status": "unsafe",
            "reason": (f"Intermediates cannot clear this much water "
                       f"(wetness {wetness:.0f}). Aquaplaning risk."),
        }

    if wetness < band["safe_min"]:
        return {
            "status": "degrading",
            "reason": (f"{'Full wets' if current_tire == 'FULL_WET' else 'Intermediates'}"
                       f" have no water left to cool them at wetness "
                       f"{wetness:.0f} — they will overheat and go off."),
        }

    return {"status": "ok", "reason": None}


def slick_compound(laps_remaining: int, severity: str) -> str:
    """Which slick, not just 'slicks'.

    Short stint -> soft; long stint -> hard. Abrasive circuits shift one step
    harder, because severity shortens every compound's usable life.
    """
    if laps_remaining <= COMPOUND_LAPS_SOFT:
        choice = "SOFT"
    elif laps_remaining <= COMPOUND_LAPS_MEDIUM:
        choice = "MEDIUM"
    else:
        choice = "HARD"

    if SEVERITY_FACTOR.get(severity, 1.0) >= 1.15:
        choice = {"SOFT": "MEDIUM", "MEDIUM": "HARD", "HARD": "HARD"}[choice]
    return choice


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
def recommend(*, label: str, trend: str, wetness: float, confidence: float,
              circuit_key: str = "silverstone",
              current_tire: str = "INTER", tire_age: int = 0,
              current_lap: int = 0, total_laps: int = 0,
              compounds_used: list[str] | None = None,
              track_temp: float | None = None, air_temp: float | None = None,
              humidity: float | None = None,
              wind_speed: float | None = None) -> dict:

    circuit = circuits().get(circuit_key, {})

    # Fall back to the circuit's typical race-day conditions rather than one
    # global default. Bahrain at 29C/45% and Spa at 29C/72% dry at very
    # different rates, and a single fixed default made the physics decorative.
    tw = circuit.get("typical_weather", {})
    weather_source = "supplied"
    if track_temp is None:
        track_temp = tw.get("track_temp", 25.0)
        weather_source = "circuit_typical"
    if air_temp is None:
        air_temp = tw.get("air_temp", 18.0)
    if humidity is None:
        humidity = tw.get("humidity", 60.0)
    if wind_speed is None:
        wind_speed = tw.get("wind_speed", 2.0)
    lap_time = float(circuit.get("lap_time_s", 90.0))
    pit_loss = float(circuit.get("pit_loss_s", 21.0))
    severity = circuit.get("severity", "medium")

    # Accept a comma-separated string as well as a list. Iterating a bare
    # string yields CHARACTERS, so "MEDIUM" would silently become
    # ['M','E','D','I','U','M'] and match no compound at all - a bug that
    # reports "0 compounds used" while looking like it worked.
    if isinstance(compounds_used, str):
        compounds_used = [c for c in compounds_used.split(",") if c]
    compounds_used = [c.strip().upper() for c in (compounds_used or [])]

    laps_remaining = max(total_laps - current_lap, 0) if total_laps else 0

    dry = drying_rate(circuit, track_temp, air_temp, humidity, wind_speed)
    rate = dry["rate_per_min"]

    family = optimal_family(wetness)
    if family == "SLICK":
        suggested = slick_compound(laps_remaining or 40, severity)
    else:
        suggested = family

    crossover = laps_to_target(wetness, DRY_THRESHOLD, rate, lap_time)

    # A crossover that arrives after the flag is not a strategy call.
    # Monaco in a downpour projected 78 laps - arithmetically right, and
    # useless. Report it as unreachable instead of quoting a number nobody
    # can act on.
    crossover_reachable = (crossover is not None and laps_remaining > 0
                           and crossover <= laps_remaining)
    if crossover is not None and laps_remaining > 0 and not crossover_reachable:
        crossover = None

    notes: list[str] = []
    urgency = "INFO"
    window_worth_taking = None

    # Is the tyre currently fitted usable at all? This runs first and can
    # pre-empt everything below - a hazard is not negotiable against lap time.
    fitness = assess_current_tire(current_tire, wetness)

    # ---- the headline ----
    if family == "FULL_WET":
        urgency = "URGENT"
        message = "Standing water. Full wets."
    elif family == "INTER":
        message = "Intermediates."
        if trend == "DRYING" and crossover is not None:
            # A window that opens is not automatically a window worth taking.
            # Check the stop it would imply: crossing over with two laps left
            # means paying a full pit loss to recover almost nothing.
            laps_after = laps_remaining - crossover if laps_remaining else None
            if laps_after is not None and \
                    laps_after * TIME_GAIN_SLICK_VS_INTER < pit_loss:
                message = (f"Slick window opens in ~{crossover} laps, but only "
                           f"{laps_after} lap{'s' if laps_after != 1 else ''} "
                           f"would remain — not worth the stop. Stay out.")
                window_worth_taking = False
            else:
                urgency = "ADVISORY"
                message = (f"Track drying. Slick window in ~{crossover} lap"
                           f"{'s' if crossover != 1 else ''}.")
        elif trend == "DRYING":
            message = ("Track drying, but the slick window will not open "
                       "before the flag. Stay on intermediates.")
    else:
        message = f"Slicks optimal — {suggested}."

    # ---- intermediates overheat once the track dries ----
    # State-based, not temperature-based: inters are cooled by the water they
    # displace, so a drying line removes the cooling and the tread goes off
    # in a handful of laps. This is the rule that makes DRYING urgent rather
    # than merely interesting.
    if (current_tire == "INTER" and trend == "DRYING" and wetness < 65
            and window_worth_taking is not False):
        urgency = "URGENT"
        notes.append(f"Inters are {tire_age} laps old with no water left to "
                     f"cool them — they will overheat.")

    # ---- full wets on a drying track ----
    if current_tire == "FULL_WET" and wetness < INTER_THRESHOLD + 10:
        urgency = "URGENT"
        notes.append("Full wets on a track this dry will overheat and blister.")

    # ---- rain arriving ----
    if trend == "WETTING" and current_tire in ("SOFT", "MEDIUM", "HARD"):
        urgency = "URGENT"
        # Name the tyre actually being suggested. Hardcoding "intermediates"
        # here contradicted a FULL_WET recommendation on a flooded track -
        # and the test suite missed it, because it checked the urgency and
        # the suggested tyre but never that the MESSAGE agreed with them.
        target = {"FULL_WET": "full wets", "INTER": "intermediates"}.get(
            family, "wet tyres")
        message = f"Wetness rising on slicks. Box for {target}."

    # ---- cold tires ----
    # Blanket limits mean a fresh slick arrives cold, and a drying track
    # offers little to heat it with. Phrased qualitatively - no number here
    # is worth defending.
    if family == "SLICK" and trend == "DRYING":
        notes.append("Fresh slicks will be cold — expect low grip on the out lap.")

    # ---- two-compound rule ----
    # Waived the moment any wet-weather tire is used, which is why a wet race
    # can legally run to the flag with no stop at all.
    wets_used = any(c in ("INTER", "FULL_WET") for c in compounds_used) \
        or current_tire in ("INTER", "FULL_WET")
    slicks_used = {c for c in compounds_used if c in ("SOFT", "MEDIUM", "HARD")}
    if wets_used:
        rule_note = "Two-compound requirement waived — wet tires used."
    elif len(slicks_used) < 2:
        rule_note = (f"Two-compound rule: only {len(slicks_used)} slick "
                     f"compound used. A second is still required.")
    else:
        rule_note = "Two-compound requirement satisfied."

    # ---- endgame: is the stop worth making? ----
    # Late in a race the pit loss can exceed everything you would win back.
    # This is the rule that says STAY OUT when the tire is technically wrong.
    #
    # Only evaluated for a change of FAMILY (wet <-> dry). A slick-to-slick
    # swap is worth nothing like 2.5s/lap, so applying this gain to a
    # MEDIUM -> HARD change would be modelling nonsense dressed as a
    # calculation.
    stop_worth_it = None
    current_family = ("SLICK" if current_tire in ("SOFT", "MEDIUM", "HARD")
                      else current_tire)
    family_change = family != current_family

    # Economics apply to an OPPORTUNITY, never to a hazard. If the tyre on
    # the car cannot handle the water, the stop happens regardless of what it
    # costs - which is exactly the check that was missing.
    if fitness["status"] in ("unsafe", "degrading"):
        pass
    elif laps_remaining > 0 and family_change:
        gain = laps_remaining * TIME_GAIN_SLICK_VS_INTER
        stop_worth_it = gain >= pit_loss
        if not stop_worth_it:
            urgency = "INFO"
            message = (f"STAY OUT — {laps_remaining} laps left. Stop costs "
                       f"~{pit_loss:.0f}s, you would recover only ~{gain:.0f}s.")
            suggested = current_tire
            notes.clear()

    # ---- PRECEDENCE 1: safety overrides everything ----
    if fitness["status"] == "unsafe":
        urgency = "URGENT"
        target = {"FULL_WET": "full wets", "INTER": "intermediates"}.get(
            family, "wet tyres")
        message = f"UNSAFE TYRE — box immediately for {target}."
        suggested = family
        notes.insert(0, fitness["reason"])
        stop_worth_it = True          # never traded against lap time
    elif fitness["status"] == "degrading" and fitness["reason"] not in notes:
        urgency = "URGENT"
        notes.insert(0, fitness["reason"])

    if confidence < 0.45:
        notes.append("Low classifier confidence — treat this as provisional.")

    return {
        "suggested_tire": suggested,
        "current_tire": current_tire,
        "change_needed": suggested != current_tire,
        "urgency": urgency,
        "message": message,
        "notes": notes,
        "rule_note": rule_note,
        "laps_to_crossover": crossover,
        "laps_remaining": laps_remaining or None,
        "stop_worth_it": stop_worth_it,
        "tire_fitness": fitness["status"],
        "window_worth_taking": window_worth_taking,
        "drying": dry,
        "weather": {
            "track_temp": track_temp, "air_temp": air_temp,
            "humidity": humidity, "wind_speed": wind_speed,
            "source": weather_source,
        },
        "circuit": {
            "key": circuit_key,
            "name": circuit.get("name", circuit_key),
            "lap_time_s": lap_time,
            "pit_loss_s": pit_loss,
            "severity": severity,
        },
    }
