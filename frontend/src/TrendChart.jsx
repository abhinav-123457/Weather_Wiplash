import React from "react";

/**
 * TrendChart - the hero component. Hand-drawn SVG rather than a chart
 * library: no dependency, full control, and it renders identically offline.
 *
 * Three things are plotted deliberately:
 *   threshold bands   the wetness regions the label thresholds define
 *   faint dots        raw per-frame scores
 *   bold line         EMA-smoothed track, coloured by the current call
 *
 * The bands matter as much as the line. Without them "56.8" is a number
 * with no meaning; with them a viewer sees instantly that the track is in
 * the damp region and how far it is from the dry one - which is the actual
 * question a race engineer is asking.
 *
 * Showing raw AND smoothed proves the temporal layer is doing real work.
 * A single smooth line asks the viewer to take that on faith.
 */

// Mirrors the flag semantics in styles.css. Duplicated as literals because
// SVG presentation attributes cannot read CSS custom properties.
const COND = {
  DRY: "#2ed573", DAMP: "#ffc300", WET: "#ff2d2d", DRYING: "#00c2ff",
};
const GRID = "#262d38";
const FAINT = "#5f6b7b";
const DIM = "#97a3b2";
const BG = "#0a0c0f";

// Mirrors DRY_THRESHOLD / DAMP_THRESHOLD in backend/app/config.py.
const T_DRY = 45, T_DAMP = 65;

export default function TrendChart({ data, labels }) {
  // Hover state, so individual frames can be inspected. A trend line you
  // cannot interrogate is a picture, not a readout - and the per-frame
  // numbers are exactly what you need when a point looks wrong.
  const [hover, setHover] = React.useState(null);
  const svgRef = React.useRef(null);

  const W = 620, H = 250;
  const PAD = { l: 32, r: 12, t: 10, b: 24 };
  const iw = W - PAD.l - PAD.r;
  const ih = H - PAD.t - PAD.b;

  if (!data || data.length === 0) {
    return <div className="empty">no frames yet — upload to begin</div>;
  }

  const n = data.length;
  const x = (i) => PAD.l + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v) => PAD.t + ih - (Math.max(0, Math.min(100, v)) / 100) * ih;

  const line = data
    .map((d, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(d.smooth).toFixed(1)}`)
    .join(" ");
  const area =
    `${line} L${x(n - 1).toFixed(1)},${PAD.t + ih} L${x(0).toFixed(1)},${PAD.t + ih} Z`;

  // Vertical rules where the committed label actually changed - the only
  // moments a race engineer would act on.
  const changes = [];
  for (let i = 1; i < (labels?.length || 0); i++) {
    if (labels[i] !== labels[i - 1]) changes.push({ i, label: labels[i] });
  }

  // The line takes the colour of the CURRENT call, so the chart carries the
  // state too - glance at it and the trend and the verdict arrive together.
  const now = labels?.[n - 1];
  const lineColor = COND[now] || DIM;

  const onMove = (e) => {
    const r = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let best = 0, bestD = Infinity;
    for (let i = 0; i < n; i++) {
      const d = Math.abs(x(i) - px);
      if (d < bestD) { bestD = d; best = i; }
    }
    setHover(best);
  };

  const bands = [
    { from: 0, to: T_DRY, color: COND.DRY, label: "DRY" },
    { from: T_DRY, to: T_DAMP, color: COND.DAMP, label: "DAMP" },
    { from: T_DAMP, to: 100, color: COND.WET, label: "WET" },
  ];

  return (
    <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
         aria-label="Track wetness over time"
         onPointerMove={onMove} onPointerLeave={() => setHover(null)}
         style={{ touchAction: "none", display: "block" }}>
      <defs>
        <linearGradient id="fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={lineColor} stopOpacity="0.28" />
          <stop offset="100%" stopColor={lineColor} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Threshold bands - what the numbers on the axis MEAN. */}
      {bands.map((b) => (
        <g key={b.label}>
          <rect x={PAD.l} y={y(b.to)} width={iw} height={y(b.from) - y(b.to)}
                fill={b.color} opacity="0.06" />
          <text x={PAD.l + 5} y={y(b.to) + 11} fontSize="8" fill={b.color}
                opacity="0.75" fontFamily="ui-monospace, monospace"
                letterSpacing="1.5">
            {b.label}
          </text>
        </g>
      ))}

      {/* Threshold lines, drawn over the bands so the boundary is exact. */}
      {[T_DRY, T_DAMP].map((v) => (
        <line key={v} x1={PAD.l} y1={y(v)} x2={W - PAD.r} y2={y(v)}
              stroke={GRID} strokeWidth="1" strokeDasharray="2 4" />
      ))}

      {[0, 25, 50, 75, 100].map((v) => (
        <text key={v} x={PAD.l - 6} y={y(v) + 3} textAnchor="end"
              fontSize="9" fill={FAINT} fontFamily="ui-monospace, monospace">
          {v}
        </text>
      ))}

      {changes.map((c) => (
        <line key={c.i} x1={x(c.i)} y1={PAD.t} x2={x(c.i)} y2={PAD.t + ih}
              stroke={COND[c.label] || DIM} strokeWidth="1.5"
              strokeDasharray="3 3" opacity="0.8" />
      ))}

      <path d={area} fill="url(#fill)" />
      <path d={line} fill="none" stroke={lineColor} strokeWidth="2.5"
            strokeLinejoin="round" strokeLinecap="round" />

      {data.map((d, i) => (
        <circle key={i} cx={x(i)} cy={y(d.raw)} r="2"
                fill={DIM} opacity="0.5" />
      ))}

      {/* Emphasised endpoint - where the track is right now. */}
      <circle cx={x(n - 1)} cy={y(data[n - 1].smooth)} r="4.5"
              fill={lineColor} stroke={BG} strokeWidth="2" />

      {hover !== null && (
        <g>
          <line x1={x(hover)} y1={PAD.t} x2={x(hover)} y2={PAD.t + ih}
                stroke={DIM} strokeWidth="1" opacity="0.45" />
          <circle cx={x(hover)} cy={y(data[hover].raw)} r="3.5" fill={DIM} />
          <circle cx={x(hover)} cy={y(data[hover].smooth)} r="4"
                  fill={lineColor} stroke={BG} strokeWidth="1.5" />
          {/* Flip the readout near the right edge so it never runs off. */}
          <g transform={`translate(${x(hover) + (hover > n * 0.66 ? -130 : 8)}, ${PAD.t + 6})`}>
            <rect width="122" height="54" fill={BG} stroke={GRID}
                  opacity="0.97" />
            <rect width="3" height="54" fill={COND[labels?.[hover]] || DIM} />
            <text x="10" y="16" fontSize="9" fill={FAINT}
                  fontFamily="ui-monospace, monospace">
              frame {hover + 1}{labels?.[hover] ? ` · ${labels[hover]}` : ""}
            </text>
            <text x="10" y="31" fontSize="10" fill={lineColor}
                  fontFamily="ui-monospace, monospace">
              smooth {data[hover].smooth.toFixed(1)}
            </text>
            <text x="10" y="45" fontSize="10" fill={DIM}
                  fontFamily="ui-monospace, monospace">
              raw    {data[hover].raw.toFixed(1)}
            </text>
          </g>
        </g>
      )}

      <text x={PAD.l} y={H - 6} fontSize="9" fill={FAINT}
            fontFamily="ui-monospace, monospace">frame 1</text>
      <text x={W - PAD.r} y={H - 6} textAnchor="end" fontSize="9" fill={FAINT}
            fontFamily="ui-monospace, monospace">frame {n}</text>
    </svg>
  );
}