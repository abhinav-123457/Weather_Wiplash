import React, { useEffect, useState } from "react";

/**
 * RaceInputs - the two things a camera cannot see and a pit wall truly
 * knows: which circuit this is, and what tyre the car is on.
 *
 * Weather is no longer typed in. Four number boxes full of guessed values
 * were indefensible - "how do you know it's 32°C?" has no good answer when
 * the answer is "we made it up". Now the backend fetches the circuit's
 * ACTUAL current weather (Open-Meteo, labelled as live) and this panel just
 * shows it. If the feed is unreachable, it says so instead of pretending.
 */

// Families, not compounds. Choosing between soft/medium/hard needs
// degradation modelling the brief does not ask for and this project cannot
// support honestly.
const TIRES = [
  ["SLICK", "Slicks"], ["INTER", "Intermediates"], ["FULL_WET", "Full wets"],
];

export default function RaceInputs({ v, set, circuits }) {
  const [wx, setWx] = useState(null);

  useEffect(() => {
    let dead = false;
    setWx(null);
    fetch(`/api/weather/${v.circuit}`)
      .then((r) => r.json())
      .then((j) => { if (!dead) setWx(j); })
      .catch(() => { if (!dead) setWx({ source: "offline" }); });
    return () => { dead = true; };
  }, [v.circuit]);

  const w = wx?.weather;
  const live = wx?.source === "live";
  const raining = live && w?.rain_mm > 0;

  return (
    <div className="panel">
      <h2>Race context</h2>

      <div className="fieldrow">
        <label className="field" style={{ minWidth: 170 }}>
          <span className="cap">circuit</span>
          <select value={v.circuit}
                  onChange={(e) => set("circuit", e.target.value)}>
            {Object.entries(circuits || {}).map(([k, c]) => (
              <option key={k} value={k}>
                {c.name || k}{c.country ? ` (${c.country})` : ""}
              </option>
            ))}
          </select>
        </label>

        <label className="field" style={{ minWidth: 150 }}>
          <span className="cap">current tyre</span>
          <select value={v.current_tire}
                  onChange={(e) => set("current_tire", e.target.value)}>
            {TIRES.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </select>
        </label>

        <div className="field">
          <span className="cap">weather at circuit</span>
          <div className="controls" style={{ margin: 0 }}>
            {wx == null ? (
              <span className="pill">fetching…</span>
            ) : live ? (
              <>
                <span className="pill">
                  {w.air_temp}°C · {w.humidity}% RH · wind {w.wind_speed} m/s
                </span>
                <span className={`pill${raining ? " warn" : ""}`}>
                  rain {w.rain_mm} mm/h
                </span>
                <span className="pill">live · Open-Meteo</span>
              </>
            ) : (
              <span className="pill warn">
                live feed unreachable — typical values in use
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}