import React from "react";

/**
 * The suggestion message - one of the three outputs the brief names:
 *
 *   "a label for each image, a simple trend graph, and a suggestion message
 *    (e.g. 'Track drying: tire change window approaching')"
 *
 * Deliberately small. An earlier version showed compound choice, tyre age,
 * the two-compound rule and pit-loss economics - none of which the brief
 * asks for, and each of which was another claim that had to be defended.
 *
 * `embedded` drops the panel chrome so this can sit directly under the
 * label and above the trend chart. As its own panel it landed below the
 * fold, and a suggestion nobody scrolls to is a suggestion nobody reads.
 */

const FAMILY = {
  SLICK: { label: "Slicks", dot: "#e8e8e8" },
  INTER: { label: "Intermediates", dot: "#43b02a" },
  FULL_WET: { label: "Full wets", dot: "#0067b1" },
};

const ICON = { INFO: "i", ADVISORY: "!", URGENT: "!!" };

function Tyre({ name }) {
  const f = FAMILY[name];
  if (!f) return <span>—</span>;
  return (
    <span className="tyre">
      <i style={{ background: f.dot }} />{f.label}
    </span>
  );
}

export default function StrategyPanel({ rec, embedded = false }) {
  const title = embedded
    ? <span className="cap">suggestion</span>
    : <h2>Suggestion</h2>;

  const body = !rec ? (
    <div className="empty">upload a frame to get a suggestion</div>
  ) : (
    <>
      {/* Urgency is carried in colour as well as text - a pit wall is
          scanned, not read. */}
      <div className={`banner ${rec.urgency || "INFO"}`}>
        <span className="badge">
          {ICON[rec.urgency || "INFO"]} {rec.urgency || "INFO"}
        </span>
        <div className="msgwrap">
          <span className="msg">{rec.message || rec.headline}</span>
          {/* The rules-decided line stays visible under any generated one,
              so what the system DECIDED is never hidden by how it was
              phrased. */}
          {rec.message_source === "model" && (
            <span className="sub">
              {rec.headline}{rec.detail ? ` — ${rec.detail}` : ""}
            </span>
          )}
          {rec.message_source !== "model" && rec.detail && (
            <span className="sub">{rec.detail}</span>
          )}
        </div>
      </div>

      {rec.notes?.length > 0 && (
        <ul className="notes">
          {rec.notes.map((n, i) => <li key={i}>{n}</li>)}
        </ul>
      )}

      {/* The slick-window and drying-rate tiles were removed on purpose:
          both rested on a tuning constant, and a number nobody can defend
          on stage is worse than no number. The tyre call is the product. */}
      <div className="stats">
        <div className="stat">
          <span className="cap">tyres</span>
          <span className="val">
            <Tyre name={rec.current_tire} />
            {rec.change_needed && (
              <>
                <span className="arrow">→</span>
                <Tyre name={rec.suggested_tire} />
              </>
            )}
          </span>
        </div>
      </div>

      {/* Nothing on screen should be unexplainable - this is what the
          sentence above was derived from. */}
      <div className="controls">
        <span className="pill">from: {rec.basis}</span>
        {rec.message_source === "model" && (
          <span className="pill">wording: flan-t5</span>
        )}
        {rec.weather && (
          <span className="pill">
            {rec.weather.air_temp}°C · {rec.weather.humidity}% RH
            {rec.weather.rain_mm > 0 && ` · rain ${rec.weather.rain_mm} mm/h`}
            {" · "}{rec.weather.source}
          </span>
        )}
      </div>
    </>
  );

  return (
    <div className={embedded ? "embedded-block" : "panel"}>
      {title}
      {body}
    </div>
  );
}