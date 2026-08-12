"""
weather.py - live conditions at the circuit, from Open-Meteo.

WHY LIVE DATA
-------------
The suggestion layer previously fell back to per-circuit "typical race-day
conditions" - a guessed 32C/75% shown on screen next to real measurements.
Nobody could defend that number, because it was not a measurement of
anything. Open-Meteo is free, needs no API key, and answers the only
defendable way: the actual current weather at the circuit's coordinates,
labelled as what it is.

HONESTY RULES
  - every response downstream carries source = operator / live / typical,
    and the UI shows which one it is displaying
  - if the API is unreachable, the system falls back to typical values AND
    SAYS SO, rather than failing or silently pretending
  - one fetch per circuit per WEATHER_TTL_S; a failure is remembered for
    60s so an offline demo machine never pays a 4-second timeout per frame
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from ..config import USE_LIVE_WEATHER, WEATHER_TTL_S

# (lat, lon) -> (fetched_at, payload | None). None records a failed fetch.
_cache: dict[tuple, tuple[float, dict | None]] = {}
_FAILURE_TTL_S = 60.0


def live(lat, lon) -> dict | None:
    """Current weather at the coordinates, or None if unavailable."""
    if not USE_LIVE_WEATHER or lat is None or lon is None:
        return None

    key = (round(float(lat), 2), round(float(lon), 2))
    now = time.time()
    hit = _cache.get(key)
    if hit is not None:
        age, payload = now - hit[0], hit[1]
        if age < (WEATHER_TTL_S if payload is not None else _FAILURE_TTL_S):
            return payload

    params = urllib.parse.urlencode({
        "latitude": key[0], "longitude": key[1],
        "current": ("temperature_2m,relative_humidity_2m,precipitation,"
                    "rain,wind_speed_10m,cloud_cover"),
        # Open-Meteo defaults to km/h; the physics downstream expects m/s.
        "wind_speed_unit": "ms",
    })
    url = f"https://api.open-meteo.com/v1/forecast?{params}"

    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            cur = json.loads(r.read().decode())["current"]
        payload = {
            "air_temp": float(cur["temperature_2m"]),
            "humidity": float(cur["relative_humidity_2m"]),
            "wind_speed": float(cur["wind_speed_10m"]),
            "rain_mm": float(cur.get("precipitation", 0.0)),
            "cloud_cover": float(cur.get("cloud_cover", 0.0)),
            "provider": "open-meteo",
            "fetched_unix": int(now),
        }
    except Exception:
        payload = None          # offline / timeout - remembered for 60s

    _cache[key] = (now, payload)
    return payload