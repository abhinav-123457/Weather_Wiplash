import React from "react";

/**
 * WhyPanel - the reasoning behind the current call.
 *
 * Every value shown here comes from the backend, computed by the same code
 * that made the decision. Nothing is re-derived in the browser: a panel that
 * reasons independently will eventually describe a decision the backend did
 * not make, and an explanation that can be wrong is worse than none.
 *
 * Three things this deliberately never claims:
 *   - "dry line detected"     the feature was measured and rejected
 *   - "the track is N% dry"   the score is not calibrated that way
 *   - certainty it does not have
 *
 * It reports the checks that actually ran, including the ones that failed -
 * knowing why the system is NOT calling a trend is as useful as knowing why
 * it is.
 */

const COLOR = {
  DRY: "var(--dry)", DAMP: "var(--damp)",
  WET: "var(--wet)", DRYING: "var(--drying)",
};

export default function WhyPanel({ frame, rec }) {
  if (!frame) {
    return (
      <div className="panel">
        <h2>Why this call</h2>
        <div className="empty">analyse a frame to see the reasoning</div>
      </div>
    );
  }

  const ev = frame.evidence || [];
  const arrow = frame.slope > 0 ? "▲" : frame.slope < 0 ? "▼" : "—";

  return (
    <div className="panel">
      <h2>Why this call</h2>

      <div className="why-head">
        <span className="why-label" style={{ color: COLOR[frame.label] }}>
          {frame.label}
        </span>
        <span className="why-since">
          held {frame.label_frames} frame{frame.label_frames === 1 ? "" : "s"}
        </span>
      </div>

      {/* What was seen this frame, before any smoothing. */}
      <div className="why-block">
        <span className="cap">visual signal</span>
        <div className="why-rows">
          <div><span>this frame</span><b>{frame.wetness_raw?.toFixed(1)}</b></div>
          <div><span>smoothed</span><b>{frame.wetness?.toFixed(1)}</b></div>
          <div><span>confidence</span><b>{frame.confidence?.toFixed(2)}</b></div>
          {/* Present only when the operator marked a dry reference - the
              per-camera offset the backend subtracted from this reading. */}
          {frame.reference_offset != null && (
            <div>
              <span>dry reference</span>
              <b>
                {frame.reference_offset > 0 ? "−" : "+"}
                {Math.abs(frame.reference_offset).toFixed(1)}
                {" "}(read {frame.wetness_uncalibrated?.toFixed(1)})
              </b>
            </div>
          )}
          {/* Second opinion from the public-road probe (1M images, zero F1
              frames). Display only - it has a measured night-race blind
              spot, so it never votes. Disagreement is worth a glance, not
              an alarm. */}
          {frame.road_opinion && (
            <div>
              <span>road model (2nd opinion)</span>
              <b style={{ color: COLOR[frame.road_opinion.label] }}>
                {frame.road_opinion.label.toLowerCase()}
                {" "}· {frame.road_opinion.wetness?.toFixed(0)}
              </b>
            </div>
          )}
        </div>
      </div>

      {/* How it is moving. */}
      <div className="why-block">
        <span className="cap">temporal evidence</span>
        <div className="why-rows">
          <div><span>trend</span><b>{frame.trend}</b></div>
          <div>
            <span>slope</span>
            <b>{arrow} {Math.abs(frame.slope ?? 0).toFixed(1)} / frame</b>
          </div>
          <div>
            <span>sustained</span>
            <b>{frame.trend_frames} frame{frame.trend_frames === 1 ? "" : "s"}</b>
          </div>
        </div>
      </div>

      {/* The checks themselves, straight from the backend. */}
      <div className="why-block">
        <span className="cap">checks</span>
        <ul className="checklist">
          {ev.map((e, i) => (
            <li key={i} className={e.pass ? "ok" : "no"}>
              <span className="mark">{e.pass ? "✓" : "✕"}</span>
              <span className="txt">
                {e.check}
                <em>{e.detail}</em>
              </span>
            </li>
          ))}
        </ul>
      </div>

      {/* The consequence - what this means for the tyre call, which is the
          only part a race engineer actually acts on. */}
      {rec && (
        <div className="why-block">
          <span className="cap">consequence</span>
          <div className="why-rows">
            <div>
              <span>suggested</span>
              <b>{(rec.suggested_tire || "—").replace("_", " ")}</b>
            </div>
            {rec.safety && rec.safety !== "ok" && (
              <div><span>tyre fitness</span><b>{rec.safety}</b></div>
            )}
            {/* Live rain is a weather-station fact, not a pixel guess -
                the one weather row that earns a place here. */}
            {rec.weather?.rain_mm > 0 && (
              <div>
                <span>rain at circuit</span>
                <b>{rec.weather.rain_mm} mm/h (live)</b>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}