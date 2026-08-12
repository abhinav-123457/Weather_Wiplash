#!/usr/bin/env python3
"""
refresh_circuits.py — replace estimated lap times and pit losses in
circuits.json with values measured from real F1 sessions via FastF1.

WHY THIS IS A BUILD-TIME SCRIPT, NOT A RUNTIME DEPENDENCY
---------------------------------------------------------
You run this once. It rewrites circuits.json with measured numbers and you
commit the result. The application never imports FastF1, never makes a network
call, and carries no extra demo risk. Install FastF1 in a SEPARATE venv so
pandas and matplotlib never enter the app's runtime footprint:

    python3 -m venv .tools-venv
    source .tools-venv/bin/activate
    pip install fastf1

USAGE
-----
    python tools/refresh_circuits.py --dry-run     # preview, write nothing
    python tools/refresh_circuits.py               # update circuits.json
    python tools/refresh_circuits.py --year 2024   # pin a season
    python tools/refresh_circuits.py --only monaco spa

WHAT IT TOUCHES
---------------
Updates ONLY:   lap_time_s, pit_loss_s
Preserves:      name, country, surface_type, shade, climate, severity, notes

Those preserved fields are hand-authored judgments (see circuits.json _meta).
Nothing here should overwrite them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    import fastf1
    import pandas as pd
except ImportError:
    sys.exit(
        "FastF1 is not installed in this interpreter.\n"
        "Use a separate venv so it stays out of the app runtime:\n"
        "    python3 -m venv .tools-venv && source .tools-venv/bin/activate\n"
        "    pip install fastf1"
    )


# --------------------------------------------------------------------------
# Circuit key -> (lookup query, required substring of the resulting Location)
#
# USE LOCATIONS, NOT COUNTRIES. get_session() fuzzy-matches, and country names
# match disastrously: "Great Britain" resolves to the AUSTRIAN Grand Prix at
# Spielberg. That produced a 70.7s "Silverstone" lap - a perfectly normal Red
# Bull Ring time, which is exactly why it looked like real data instead of an
# error. Circuit locations are unambiguous.
#
# The second element is checked against the event FastF1 actually returned.
# A mismatch aborts that circuit rather than writing plausible-looking garbage.
# --------------------------------------------------------------------------
FASTF1_EVENT = {
    "bahrain":     ("Sakhir",            "Sakhir"),
    "jeddah":      ("Jeddah",            "Jeddah"),
    "suzuka":      ("Suzuka",            "Suzuka"),
    "miami":       ("Miami",             "Miami"),
    "monaco":      ("Monaco",            "Monaco"),
    "montreal":    ("Montréal",          "Montr"),
    "silverstone": ("Silverstone",       "Silverstone"),
    "spa":         ("Spa-Francorchamps", "Spa"),
    "zandvoort":   ("Zandvoort",         "Zandvoort"),
    "singapore":   ("Marina Bay",        "Marina Bay"),
}

# Seasons to try, newest first. If a circuit was not on a given calendar the
# lookup fails and we fall through to the next year.
CANDIDATE_YEARS = [2025, 2024, 2023]

CACHE_DIR = Path(".fastf1-cache")


# --------------------------------------------------------------------------
# Lap filtering
# --------------------------------------------------------------------------
def green_flag_laps(laps: "pd.DataFrame") -> "pd.DataFrame":
    """
    Representative racing laps only.

    Filtered deliberately rather than via pick_* helpers, whose names have
    shifted between FastF1 versions. Column filtering is version-stable.

      TrackStatus == "1"  -> all clear (no yellow, SC, VSC, red)
      IsAccurate          -> FastF1's own timing-quality flag
      no in/out laps      -> pit laps are not representative pace
      LapNumber > 1       -> lap 1 includes the standing start
    """
    clean = laps[
        laps["LapTime"].notna()
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & (laps["LapNumber"] > 1)
    ]

    if "TrackStatus" in clean.columns:
        clean = clean[clean["TrackStatus"].astype(str) == "1"]

    if "IsAccurate" in clean.columns:
        clean = clean[clean["IsAccurate"]]

    return clean


def median_lap_time_s(laps: "pd.DataFrame") -> float | None:
    clean = green_flag_laps(laps)
    if len(clean) < 20:          # too thin to trust
        return None
    return round(clean["LapTime"].median().total_seconds(), 1)


# --------------------------------------------------------------------------
# Pit loss
# --------------------------------------------------------------------------
def pit_loss_s(laps: "pd.DataFrame") -> tuple[float | None, int]:
    """
    Total time lost to a pit stop, measured against the driver's own pace.

        loss = (in-lap + out-lap) - 2 x that driver's median green-flag lap

    Using a per-driver baseline matters: a backmarker's out-lap should be
    compared against a backmarker's normal lap, not the field median, or
    slower teams inflate the result.

    Returns (median_loss_seconds, number_of_stops_sampled).
    """
    losses: list[float] = []

    for drv, drv_laps in laps.groupby("Driver"):
        drv_laps = drv_laps.sort_values("LapNumber")

        clean = green_flag_laps(drv_laps)
        if len(clean) < 8:
            continue
        baseline = clean["LapTime"].median().total_seconds()

        in_laps = drv_laps[drv_laps["PitInTime"].notna()]

        for _, in_lap in in_laps.iterrows():
            out_lap = drv_laps[drv_laps["LapNumber"] == in_lap["LapNumber"] + 1]
            if out_lap.empty:
                continue
            out_lap = out_lap.iloc[0]

            if pd.isna(in_lap["LapTime"]) or pd.isna(out_lap["LapTime"]):
                continue
            # Only count stops made under green - a stop under safety car
            # costs far less and would skew the median downward.
            if str(in_lap.get("TrackStatus", "1")) != "1":
                continue
            if str(out_lap.get("TrackStatus", "1")) != "1":
                continue

            combined = (in_lap["LapTime"].total_seconds()
                        + out_lap["LapTime"].total_seconds())
            loss = combined - 2 * baseline

            # Sanity window. Anything outside this is a red flag, a long
            # repair, or bad timing data - not a normal stop.
            if 10.0 < loss < 45.0:
                losses.append(loss)

    if len(losses) < 5:
        return None, len(losses)

    return round(pd.Series(losses).median(), 1), len(losses)


# --------------------------------------------------------------------------
# Per-circuit measurement
# --------------------------------------------------------------------------
def measure(key: str, years: list[int]) -> dict | None:
    mapping = FASTF1_EVENT.get(key)
    if not mapping:
        print(f"  ! no FastF1 event mapping for '{key}' - skipping")
        return None
    query, expected_location = mapping
    fallback: dict | None = None      # a wet race, used only if no dry one exists

    for year in years:
        try:
            session = fastf1.get_session(year, query, "R")

            # Guard against fuzzy-match drift BEFORE loading anything.
            # Without this, a wrong-but-plausible circuit silently supplies
            # numbers that look entirely reasonable.
            location = str(session.event.get("Location", ""))
            if expected_location.lower() not in location.lower():
                print(f"  ! {year}: '{query}' resolved to "
                      f"{session.event.get('EventName')} ({location}) - "
                      f"expected '{expected_location}'. Refusing.")
                continue

            session.load(laps=True, telemetry=False,
                         weather=True, messages=False)
        except Exception as exc:
            print(f"  · {year}: {type(exc).__name__} - {exc}")
            continue

        laps = session.laps
        if laps is None or laps.empty:
            print(f"  · {year}: no lap data")
            continue

        lap_s = median_lap_time_s(laps)
        pit_s, n_stops = pit_loss_s(laps)

        if lap_s is None:
            print(f"  · {year}: too few clean laps")
            continue

        # Was it wet? A wet race inflates lap times by 10-15% and pushes most
        # stops under safety car, leaving few green-flag samples. Mixing wet
        # and dry baselines across circuits makes the constants incoherent,
        # so prefer a dry race and only fall back to a wet one if nothing
        # else is available.
        wet = False
        try:
            wx = session.weather_data
            if wx is not None and "Rainfall" in wx:
                wet = bool(wx["Rainfall"].any())
        except Exception:
            pass

        record = {
            "lap_time_s": lap_s,
            "pit_loss_s": pit_s,
            "_source": {
                "year": year,
                "session": "Race",
                "event": str(session.event.get("EventName", query)),
                "location": location,
                "pit_stops_sampled": n_stops,
                "wet_session": wet,
            },
        }

        flag = " [WET - held as fallback]" if wet else ""
        print(f"  {'·' if wet else '✓'} {year}: lap {lap_s}s · "
              f"pit {pit_s}s ({n_stops} stops){flag}")

        if not wet:
            return record
        if fallback is None:
            fallback = record

    if fallback is not None:
        yr = fallback["_source"]["year"]
        print(f"  ! no dry race found - using wet {yr} session. "
              f"Treat these as soft.")
        return fallback

    print("  ! no usable session in any candidate year")
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="circuits.json", type=Path)
    ap.add_argument("--year", type=int,
                    help="pin a single season instead of trying newest-first")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help="update only these circuit keys")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would change, write nothing")
    args = ap.parse_args()

    if not args.file.exists():
        sys.exit(f"{args.file} not found - run from the project root.")

    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    print(f"cache: {CACHE_DIR.resolve()}\n")

    data = json.loads(args.file.read_text())
    years = [args.year] if args.year else CANDIDATE_YEARS

    keys = [k for k in data if not k.startswith("_")]
    if args.only:
        unknown = set(args.only) - set(keys)
        if unknown:
            sys.exit(f"unknown circuit key(s): {', '.join(sorted(unknown))}")
        keys = args.only

    changes: list[tuple[str, str, object, object]] = []
    measured_count = 0

    for key in keys:
        print(f"{key}:")
        result = measure(key, years)
        if not result:
            print("  → keeping existing estimates\n")
            continue

        circuit = data[key]
        for field in ("lap_time_s", "pit_loss_s"):
            new = result[field]
            if new is None:
                continue
            old = circuit.get(field)
            if old != new:
                changes.append((key, field, old, new))
            circuit[field] = new

        circuit["_measured"] = result["_source"]
        measured_count += 1
        print()

    # ---- report ----
    print("=" * 62)
    if not changes:
        print("no changes - existing values already match measurements")
    else:
        print(f"{'circuit':<14}{'field':<14}{'old':>8}{'new':>10}{'delta':>10}")
        print("-" * 62)
        for key, field, old, new in changes:
            delta = f"{new - old:+.1f}" if isinstance(old, (int, float)) else "—"
            old_s = f"{old}" if old is not None else "—"
            print(f"{key:<14}{field:<14}{old_s:>8}{new:>10}{delta:>10}")
    print("=" * 62)
    print(f"{measured_count}/{len(keys)} circuits measured")

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0

    if not changes:
        return 0

    # Provenance in _meta reflects that these are no longer guesses.
    meta = data.setdefault("_meta", {})
    prov = meta.setdefault("provenance", {})
    stamp = datetime.now().strftime("%Y-%m-%d")
    prov["lap_time_s"] = (
        f"MEASURED via FastF1 ({stamp}): median green-flag race lap, "
        "excluding pit, safety-car and opening laps."
    )
    prov["pit_loss_s"] = (
        f"MEASURED via FastF1 ({stamp}): median of "
        "(in-lap + out-lap) - 2 x driver's own median green-flag lap, "
        "green-flag stops only."
    )

    backup = args.file.with_suffix(".json.bak")
    shutil.copy2(args.file, backup)
    args.file.write_text(json.dumps(data, indent=2) + "\n")

    print(f"\nwrote {args.file}   (backup: {backup})")
    print("Review the diff, then delete the .bak once you are happy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
