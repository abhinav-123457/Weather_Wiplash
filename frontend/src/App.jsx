import React, { useCallback, useEffect, useRef, useState } from "react";
import TrendChart from "./TrendChart.jsx";
import RaceInputs from "./RaceInputs.jsx";
import StrategyPanel from "./StrategyPanel.jsx";
import WhyPanel from "./WhyPanel.jsx";

const SESSION = "ui";

// Mirrors backend ROI_PRESETS. With segmentation disabled the ROI IS the
// mask, so this is the most important control in the app. Keep these two
// lists in step - the backend uses its own copy when no roi is sent.
const PRESETS = {
  onboard: [0.08, 0.14, 0.92, 0.55],
  trackside: [0.03, 0.20, 0.97, 0.90],
};

const COLOR = {
  DRY: "var(--dry)", DAMP: "var(--damp)",
  WET: "var(--wet)", DRYING: "var(--drying)",
};

export default function App() {
  const [health, setHealth] = useState(null);
  const [camera, setCamera] = useState("trackside");
  const [frames, setFrames] = useState([]);   // one row per analysed frame
  const [latest, setLatest] = useState(null);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState(null);
  const [over, setOver] = useState(false);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  // Input mode. "stills" is the original multi-file upload; "video" samples
  // a playing video file; "camera" samples a live webcam/phone feed. The
  // brief says LIVE detector - the temporal layer was built for a stream,
  // and these two modes finally feed it one.
  const [mode, setMode] = useState("stills");
  const [liveActive, setLiveActive] = useState(false);
  const [intervalS, setIntervalS] = useState(2);
  // Auto-ROI follow: the backend re-detects the track whenever the shot
  // changes (broadcast cuts) and the drawn box syncs to what it used.
  const [autoFollow, setAutoFollow] = useState(false);
  const [videoURL, setVideoURL] = useState(null);
  const videoRef = useRef(null);
  const videoFileRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const liveCount = useRef(0);
  const inFlight = useRef(false);
  const liveSource = useRef("video");

  // The ROI is the mask - segmentation is disabled, so everything inside
  // this box is scored as track surface. A preset cannot know where the
  // track sits in an arbitrary broadcast shot, so it must be draggable.
  const [roi, setRoi] = useState(PRESETS.trackside);
  const [customRoi, setCustomRoi] = useState(false);
  const [drag, setDrag] = useState(null);
  const [testScore, setTestScore] = useState(null);
  const [append, setAppend] = useState(false);
  const [roiNote, setRoiNote] = useState(null);   // auto-detect verdict
  const [progress, setProgress] = useState(null); // batch analysis counter

  // Flash the badge when the COMMITTED label changes. That transition is
  // the entire point of the product - "the track just became wet" - and it
  // previously slipped past with no more emphasis than a number tick.
  const [flash, setFlash] = useState(false);
  const prevLabel = useRef(null);

  // Dry-reference calibration - the operator marks a frame they KNOW shows
  // dry track, and the backend subtracts the resulting offset from every
  // later frame. Per-camera answer to the measured 16-28 point venue offset.
  const [reference, setReference] = useState(null);

  // Strategy inputs. Weather fields stay null so the backend falls back to
  // the circuit's typical race-day conditions rather than one global guess.
  const [circuits, setCircuits] = useState({});
  const [race, setRace] = useState({
    circuit: "silverstone", current_tire: "INTER",
    track_temp: null, air_temp: null, humidity: null, wind_speed: null,
  });
  const setRaceField = (k, val) => setRace((r) => ({ ...r, [k]: val }));
  const [strategy, setStrategy] = useState(null);
  const lastFile = useRef(null);
  const imgRef = useRef(null);

  // The capture timer's callback closes over stale state, so the values it
  // needs live in a ref that every render refreshes. Without this, dragging
  // the ROI mid-capture would keep sampling the OLD box forever.
  const stateRef = useRef({});
  useEffect(() => { stateRef.current = { camera, roi, race, autoFollow }; });

  useEffect(() => {
    fetch("/health").then((r) => r.json()).then(setHealth).catch(() => {});
    fetch("/api/circuits").then((r) => r.json()).then(setCircuits).catch(() => {});
    return () => stopLive();          // release camera + timer on unmount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Switching camera restores that preset unless the user has drawn a box.
  useEffect(() => {
    if (!customRoi) setRoi(PRESETS[camera]);
  }, [camera, customRoi]);

  useEffect(() => {
    const l = latest?.label;
    if (!l) return;
    const changed = prevLabel.current !== null && prevLabel.current !== l;
    prevLabel.current = l;
    if (!changed) return;
    setFlash(true);
    const t = setTimeout(() => setFlash(false), 1400);
    return () => clearTimeout(t);
  }, [latest]);

  const clamp01 = (v) => Math.max(0, Math.min(1, v));

  const pointToFraction = (e) => {
    const r = imgRef.current.getBoundingClientRect();
    return [clamp01((e.clientX - r.left) / r.width),
            clamp01((e.clientY - r.top) / r.height)];
  };

  const onPointerDown = (e) => {
    if (!imgRef.current) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const [x, y] = pointToFraction(e);
    setDrag({ x0: x, y0: y, x1: x, y1: y });
  };

  const onPointerMove = (e) => {
    if (!drag) return;
    const [x, y] = pointToFraction(e);
    setDrag((d) => ({ ...d, x1: x, y1: y }));
  };

  const onPointerUp = () => {
    if (!drag) return;
    const x0 = Math.min(drag.x0, drag.x1), x1 = Math.max(drag.x0, drag.x1);
    const y0 = Math.min(drag.y0, drag.y1), y1 = Math.max(drag.y0, drag.y1);
    // Ignore accidental clicks - a box under ~4% of the frame is a misclick,
    // and an empty ROI would score nothing.
    if (x1 - x0 > 0.04 && y1 - y0 > 0.04) {
      setRoi([x0, y0, x1, y1]);
      setCustomRoi(true);
      setTestScore(null);
    }
    setDrag(null);
  };

  // Score the previewed frame in a THROWAWAY session so ROI tuning never
  // pollutes the real trend line.
  const testRoi = async () => {
    if (!lastFile.current) return;
    const fd = new FormData();
    fd.append("file", lastFile.current);
    fd.append("camera_type", camera);
    fd.append("session_id", "roi-tune");
    fd.append("roi", roi.join(","));
    await fetch("/api/session/reset",
      { method: "POST", body: (() => { const f = new FormData();
        f.append("session_id", "roi-tune"); return f; })() });
    const res = await fetch("/api/analyze/image", { method: "POST", body: fd });
    const j = await res.json();
    setTestScore(j.error ? { error: j.message || j.error } : j);
  };

  // One-shot track detection: the backend suggests a box, which lands in
  // the SAME draggable control - a bad detection costs one drag, never a
  // corrupted session.
  const autoRoi = async () => {
    let file = null;
    if (mode === "stills") {
      file = lastFile.current;
    } else {
      const vid = videoRef.current;
      if (vid && vid.readyState >= 2) {
        const c = canvasRef.current ||
          (canvasRef.current = document.createElement("canvas"));
        c.width = vid.videoWidth;
        c.height = vid.videoHeight;
        c.getContext("2d").drawImage(vid, 0, 0);
        const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", 0.85));
        if (blob) file = new File([blob], "roi_probe.jpg", { type: "image/jpeg" });
      }
    }
    if (!file) { setError("no frame on screen to detect the track from"); return; }
    setError(null);
    setRoiNote(null);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("/api/roi/suggest", { method: "POST", body: fd });
      const j = await res.json();
      if (j.error) { setError(j.message || j.error); return; }
      setRoi(j.roi);
      setCustomRoi(true);
      setTestScore(null);
      // Say what was accepted and what was thrown out - a box that appears
      // with no explanation is exactly as trustworthy as a guess.
      const n = j.rejected?.length || 0;
      setRoiNote(`verified as track (${j.verified_as?.confidence})`
        + (n ? ` — rejected ${n} region${n === 1 ? "" : "s"}: `
             + j.rejected.map((r) => r.looked_like).join(", ") : ""));
    } catch (e) {
      setError(`auto ROI: ${e.message}`);
    }
  };

  const box = drag
    ? { left: Math.min(drag.x0, drag.x1), top: Math.min(drag.y0, drag.y1),
        w: Math.abs(drag.x1 - drag.x0), h: Math.abs(drag.y1 - drag.y0) }
    : { left: roi[0], top: roi[1], w: roi[2] - roi[0], h: roi[3] - roi[1] };

  // One frame through the API, shared by the stills loop and the live
  // capture tick so the two paths cannot drift apart.
  const sendFrame = async (file, name) => {
    const { camera: cam, roi: r, race: rc, autoFollow: follow } = stateRef.current;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("camera_type", cam);
    fd.append("session_id", SESSION);
    fd.append("roi", r.join(","));
    fd.append("auto_roi", follow ? "true" : "false");
    fd.append("circuit", rc.circuit);
    fd.append("current_tire", rc.current_tire);
    // Blank weather fields are OMITTED, not sent as 0 - that is what makes
    // the backend fall back to the circuit's typical conditions.
    for (const k of ["track_temp", "air_temp", "humidity", "wind_speed"]) {
      if (rc[k] !== null && rc[k] !== undefined) fd.append(k, String(rc[k]));
    }
    try {
      const res = await fetch("/api/analyze/image", { method: "POST", body: fd });
      const j = await res.json();
      if (j.error) { setError(`${name}: ${j.message || j.error}`); return null; }
      setLatest(j);
      setFrames((prev) => [...prev, { name, ...j }]);
      // In follow mode the backend owns the box - sync the drawn one to
      // whatever it actually scored, so the screen never lies.
      if (stateRef.current.autoFollow && Array.isArray(j.roi)) setRoi(j.roi);
      return j;
    } catch (e) {
      setError(`${name}: ${e.message}`);
      return null;
    }
  };

  const resetSessionState = async () => {
    const f = new FormData();
    f.append("session_id", SESSION);
    await fetch("/api/session/reset", { method: "POST", body: f });
    setFrames([]);
    setLatest(null);
    setStrategy(null);
    setReference(null);      // backend reset cleared it - mirror that here
    prevLabel.current = null;   // next label is a first, not a change
  };

  const analyse = useCallback(async (fileList) => {
    const files = Array.from(fileList)
      .filter((f) => f.type.startsWith("image/"))
      // Filename order is frame order - the same rule the CLI runner uses.
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    if (!files.length) return;

    setBusy(true);
    setError(null);
    setTestScore(null);

    // Start a fresh session for every new batch unless explicitly appending.
    //
    // Leaving this to a button was a trap: a batch uploaded after a previous
    // run silently continued it, so frame 1 arrived pre-smoothed against the
    // old track and inherited its label. The trend line looked plausible and
    // was wrong - the worst kind of bug to have on stage.
    if (!append) await resetSessionState();

    // A counter, not a spinner: on a 20-frame batch at ~0.1s each the user
    // needs to know it is progressing, not merely that it is busy.
    setProgress({ done: 0, total: files.length });
    for (const [i, file] of files.entries()) {
      lastFile.current = file;
      setPreview(URL.createObjectURL(file));
      await sendFrame(file, file.name);
      setProgress({ done: i + 1, total: files.length });
    }
    setProgress(null);
    setBusy(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [append]);

  // ---------------- live capture ----------------
  const tick = async () => {
    const v = videoRef.current;
    if (!v || v.readyState < 2) return;
    // A paused video means the operator is scrubbing - sampling it would
    // stack identical frames and flatten the slope to zero.
    if (liveSource.current === "video" && (v.paused || v.ended)) return;
    if (inFlight.current) return;      // never queue behind a slow frame
    inFlight.current = true;
    try {
      const c = canvasRef.current ||
        (canvasRef.current = document.createElement("canvas"));
      c.width = v.videoWidth;
      c.height = v.videoHeight;
      c.getContext("2d").drawImage(v, 0, 0);
      const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", 0.85));
      if (!blob) return;
      const name = `live_${String(++liveCount.current).padStart(3, "0")}` +
        `_t${Math.round(v.currentTime)}s.jpg`;
      await sendFrame(new File([blob], name, { type: "image/jpeg" }), name);
    } finally {
      inFlight.current = false;
    }
  };

  const startLive = async (source) => {
    setError(null);
    liveSource.current = source;
    if (!append) await resetSessionState();
    liveCount.current = 0;

    if (source === "camera") {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" }, audio: false,
        });
        streamRef.current = stream;
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      } catch (e) {
        setError(`camera: ${e.message}`);
        return;
      }
    } else {
      if (!videoURL) { setError("choose a video file first"); return; }
      try { await videoRef.current.play(); } catch { /* autoplay policy */ }
    }
    setLiveActive(true);
    timerRef.current = setInterval(tick, Math.max(1, intervalS) * 1000);
    tick();
  };

  const stopLive = () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setLiveActive(false);
  };

  const switchMode = (m) => {
    stopLive();
    setMode(m);
    setTestScore(null);
  };

  // ---------------- dry reference ----------------
  const markDryReference = async () => {
    const fd = new FormData();
    fd.append("session_id", SESSION);
    const res = await fetch("/api/reference", { method: "POST", body: fd });
    const j = await res.json();
    if (j.error) setError(j.message || j.error);
    else { setReference(j.offset); setError(null); }
  };

  const clearDryReference = async () => {
    const fd = new FormData();
    fd.append("session_id", SESSION);
    fd.append("clear", "true");
    await fetch("/api/reference", { method: "POST", body: fd });
    setReference(null);
  };

  const reset = async () => {
    stopLive();
    await resetSessionState();
    setPreview(null);
    setError(null);
  };

  // Recompute the tyre call whenever the race context changes.
  //
  // Changing circuit, tyre or weather does not change what the track LOOKS
  // like, so requiring a re-upload to see the effect was wasteful and
  // confusing. Debounced so typing in a number field does not fire a request
  // per keystroke.
  useEffect(() => {
    if (!frames.length) { setStrategy(null); return; }
    const t = setTimeout(async () => {
      const fd = new FormData();
      fd.append("session_id", SESSION);
      fd.append("circuit", race.circuit);
      fd.append("current_tire", race.current_tire);
      for (const k of ["track_temp", "air_temp", "humidity", "wind_speed"]) {
        if (race[k] !== null && race[k] !== undefined) fd.append(k, String(race[k]));
      }
      try {
        const res = await fetch("/api/strategy", { method: "POST", body: fd });
        const j = await res.json();
        if (!j.error) setStrategy(j.recommendation);
      } catch { /* leave the previous call on screen */ }
    }, 250);
    return () => clearTimeout(t);
  }, [race, frames.length, latest]);

  const chart = frames.map((f) => ({ raw: f.wetness_raw, smooth: f.wetness }));
  const labels = frames.map((f) => f.label);
  const label = latest?.label ?? "—";
  const showMedia = mode === "stills" ? Boolean(preview) : true;

  return (
    <div className="app">
      {/* Sticky, and it carries the live condition. Scrolling down to the
          frame log used to lose sight of the label entirely - the one thing
          that should never leave the screen. */}
      <div className="topbar">
        <h1>Weather Whiplash</h1>

        {latest && (
          <div className={`live-state state-${label}`}>
            <span className="live-dot" />
            {label}
            <span className="live-num">{latest.wetness.toFixed(0)}</span>
          </div>
        )}

        <div className="meta">
          <span>
            <span className={health ? "dot ok" : "dot"}>●</span>{" "}
            {health ? "backend online" : "backend offline"}
          </span>
          {/* Scoring METHOD only. The probe's stored accuracy is a
              cross-validated train-set number, and an unqualified "83.3%"
              on screen contradicts the honest venue-held-out figure we
              quote everywhere else. Confirming the probe loaded is the
              useful part; the number belongs in the write-up with its
              caveat attached. */}
          {health?.scoring_method && (
            <span title="Linear probe over frozen CLIP embeddings. Honest accuracy: 85.4% wet vs not-wet, venue-held-out.">
              scoring: {health.scoring_method}
            </span>
          )}
        </div>
      </div>

      <RaceInputs v={race} set={setRaceField} circuits={circuits} />

      <div className="grid">
        {/* ---------------- left: input ---------------- */}
        <div className="panel">
          <h2>Input</h2>

          {/* Controls in three labelled groups rather than one wrapping
              row of eight. The old row read as a toolbar of equals; these
              read as the three decisions actually being made - what am I
              watching, how am I sampling it, and which session is this. */}
          <div className="ctl-group">
            <span className="ctl-label">source</span>
            <div className="controls">
              <select value={mode} onChange={(e) => switchMode(e.target.value)}>
                <option value="stills">Still frames</option>
                <option value="video">Video file</option>
                <option value="camera">Camera</option>
              </select>
              <select value={camera} onChange={(e) => setCamera(e.target.value)}>
                <option value="trackside">Trackside view</option>
                <option value="onboard">Onboard view</option>
              </select>

              {mode === "stills" && (
                <>
                  <button className="primary" onClick={() => fileRef.current?.click()}
                          disabled={busy}>
                    Choose frames
                  </button>
                  <input ref={fileRef} type="file" accept="image/*" multiple
                         hidden onChange={(e) => analyse(e.target.files)} />
                </>
              )}

              {mode === "video" && (
                <>
                  <button onClick={() => videoFileRef.current?.click()}
                          disabled={liveActive}>
                    Choose video
                  </button>
                  <input ref={videoFileRef} type="file" accept="video/*" hidden
                         onChange={(e) => {
                           const f = e.target.files[0];
                           if (f) setVideoURL(URL.createObjectURL(f));
                         }} />
                </>
              )}
            </div>
          </div>

          {mode !== "stills" && (
            <div className="ctl-group">
              <span className="ctl-label">capture</span>
              <div className="controls">
                {liveActive ? (
                  <button onClick={stopLive}>■ Stop</button>
                ) : (
                  <button className="primary" onClick={() => startLive(mode)}
                          disabled={mode === "video" && !videoURL}>
                    ▶ {mode === "camera" ? "Start camera" : "Start capture"}
                  </button>
                )}
                <label className="pill">
                  every{" "}
                  <input type="number" min={1} max={30} value={intervalS}
                         disabled={liveActive}
                         onChange={(e) => setIntervalS(Number(e.target.value) || 2)}
                         style={{ width: 40, margin: "0 4px" }} />{" "}
                  s
                </label>
                {/* EXPERIMENTAL, off by default. Measured on real Monaco
                    onboard footage it boxed the car's bodywork and a
                    driver's helmet - the camera preset plus one drag beat
                    it every time. Kept because it is the right idea for a
                    fixed trackside camera, where the view never cuts. */}
                <label className="pill" style={{ cursor: "pointer" }}
                       title="Experimental: re-detects the track on shot changes. Measured unreliable on onboard broadcast footage — the presets are better.">
                  <input type="checkbox" checked={autoFollow}
                         onChange={(e) => setAutoFollow(e.target.checked)}
                         style={{ marginRight: 6, verticalAlign: "middle" }} />
                  auto ROI (exp.)
                </label>
              </div>
            </div>
          )}

          <div className="ctl-group">
            <span className="ctl-label">session</span>
            <div className="controls">
              <button onClick={reset} disabled={busy || !frames.length}>
                New session
              </button>
              <label className="pill" style={{ cursor: "pointer" }}>
                <input type="checkbox" checked={append}
                       onChange={(e) => setAppend(e.target.checked)}
                       style={{ marginRight: 6, verticalAlign: "middle" }} />
                append to session
              </label>
              <span className="pill">{frames.length} frames</span>
            </div>
          </div>

          {mode === "stills" && (
            <div
              className={`drop${over ? " over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); analyse(e.dataTransfer.files); }}
              onClick={() => fileRef.current?.click()}
            >
              {busy && progress
                ? `analysing frame ${progress.done} of ${progress.total}…`
                : busy
                ? "analysing…"
                : "drop frames here — multiple files run in filename order"}
            </div>
          )}

          {showMedia && (
            <>
              <div className="preview-wrap"
                   onPointerDown={onPointerDown}
                   onPointerMove={onPointerMove}
                   onPointerUp={onPointerUp}
                   style={{ cursor: "crosshair", touchAction: "none" }}>
                {mode === "stills" ? (
                  <img ref={imgRef} className="preview" src={preview}
                       alt="current frame" draggable={false} />
                ) : (
                  <video ref={(el) => { videoRef.current = el; imgRef.current = el; }}
                         className="preview"
                         src={mode === "video" ? (videoURL ?? undefined) : undefined}
                         controls={mode === "video"}
                         muted playsInline
                         onEnded={stopLive} />
                )}
                {/* Drawn over the frame because it IS what gets scored -
                    showing it stops the number looking arbitrary. */}
                <div className="roi-box" style={{
                  left: `${box.left * 100}%`, top: `${box.top * 100}%`,
                  width: `${box.w * 100}%`, height: `${box.h * 100}%`,
                }} />
              </div>

              <div className="controls">
                <span className="pill">
                  drag on the image to set the scored region
                </span>
                {mode === "stills" && (
                  <button onClick={testRoi} disabled={busy}>Test ROI</button>
                )}
                <button onClick={autoRoi} disabled={busy}
                        title="Detect the track surface in the current frame and place the box on it">
                  Auto-detect track
                </button>
                <button onClick={() => { setCustomRoi(false);
                                         setRoi(PRESETS[camera]);
                                         setTestScore(null);
                                         setRoiNote(null); }}
                        disabled={!customRoi}>
                  Reset ROI
                </button>
                {customRoi && <span className="pill">custom</span>}
                {/* How much of the frame is actually being scored. The ROI
                    is the mask, so this number is the honest answer to
                    "how much track did it look at?" */}
                <span className="pill">
                  scoring {Math.round((roi[2] - roi[0]) * (roi[3] - roi[1]) * 100)}%
                  {" "}of frame
                </span>
                {roiNote && <span className="pill">{roiNote}</span>}
                {liveActive && (
                  <span className="pill">
                    ● sampling every {intervalS}s — {liveCount.current} frames
                  </span>
                )}
                {autoFollow && latest?.roi_source && (
                  <span className="pill">{latest.roi_source}</span>
                )}
              </div>

              {testScore && (
                <div className="controls">
                  {testScore.error ? (
                    <span className="pill warn">{testScore.error}</span>
                  ) : (
                    <>
                      <span className="pill" style={{ color: COLOR[testScore.state] }}>
                        {testScore.state}
                      </span>
                      <span className="pill">
                        wetness {testScore.wetness?.toFixed(1)}
                      </span>
                      <span className={`pill${testScore.confidence < 0.45 ? " warn" : ""}`}>
                        conf {testScore.confidence?.toFixed(2)}
                      </span>
                      <span className="pill">not added to session</span>
                    </>
                  )}
                </div>
              )}
            </>
          )}

          {error && <div className="pill warn">{error}</div>}
        </div>

        {/* ---------------- right: readout ---------------- */}
        <div className="panel">
          <h2>Track condition</h2>

          {/* Before any frame exists, four dashes taught a first-time
              viewer nothing. Three numbered steps get them to a result -
              and a judge opening this cold sees what it does, not what it
              lacks. */}
          {!latest ? (
            <div className="onboard-hint">
              <ol>
                <li>Pick a <b>source</b> — still frames, a race video, or a camera.</li>
                <li>Check the white box covers <b>track surface only</b>; drag to adjust.</li>
                <li>Analyse. The label, trend and tyre call appear here.</li>
              </ol>
              <span className="cap">
                dry · damp · wet from one frame — drying needs the sequence
              </span>
            </div>
          ) : (
            <>
              <div className="state-row">
                <div className={`state-badge state-${label}${flash ? " flash" : ""}`}>
                  {label}
                </div>
                <div className="readout">
                  <span className="cap">wetness</span>
                  <span className="big">{latest.wetness.toFixed(1)}</span>
                </div>
                <div className="readout">
                  <span className="cap">trend</span>
                  <span className="big" style={{ fontSize: "1rem" }}>
                    {latest.trend}
                    {latest.slope !== 0 && (
                      <span style={{ color: "var(--ink-faint)" }}>
                        {"  "}{latest.slope > 0 ? "▲" : "▼"}
                        {Math.abs(latest.slope).toFixed(1)}
                      </span>
                    )}
                  </span>
                </div>
              </div>

              <div className="gauge">
                <div style={{
                  width: `${latest.wetness}%`,
                  background: COLOR[label] || "var(--ink-faint)",
                }} />
              </div>

              <div className="controls">
                <span className="pill">state {latest.state}</span>
                <span className="pill">conf {latest.confidence.toFixed(2)}</span>
                {latest.label_held ? (
                  <span className="pill warn">
                    uncertain frame — held previous call
                  </span>
                ) : latest.low_confidence ? (
                  <span className="pill">low confidence</span>
                ) : null}
                {latest.method && <span className="pill">{latest.method}</span>}
              </div>

              {/* Per-camera dry reference. Analyse a frame that shows dry
                  track, then mark it - every later frame is corrected by
                  the offset. */}
              <div className="controls">
                <button onClick={markDryReference}
                        title="Analyse a frame you KNOW shows dry track, then click">
                  mark last frame as known dry
                </button>
                {reference != null && (
                  <>
                    <span className="pill">
                      dry ref: offset {reference > 0 ? "−" : "+"}
                      {Math.abs(reference).toFixed(1)} applied
                    </span>
                    <button onClick={clearDryReference}>clear</button>
                  </>
                )}
              </div>
            </>
          )}

          {/* The suggestion sits HERE, not in a panel further down. The
              brief names three outputs - label, trend graph, suggestion -
              and this column now carries all three above the fold, in the
              order an engineer reads them. */}
          <StrategyPanel rec={strategy || latest?.recommendation} embedded />

          <h2 style={{ marginTop: 6 }}>Trend</h2>
          <TrendChart data={chart} labels={labels} />
        </div>
      </div>

      <div className="grid">
        <WhyPanel frame={latest} rec={strategy || latest?.recommendation} />

        {/* ---------------- frame log ---------------- */}
        <div className="panel">
          <h2>Frame log</h2>
          {frames.length === 0 ? (
            <div className="empty">no frames analysed yet</div>
          ) : (
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th>file</th><th>raw</th><th>smooth</th><th>slope</th>
                    <th>state</th><th>trend</th><th>label</th><th>conf</th>
                  </tr>
                </thead>
                <tbody>
                  {frames.map((f, i) => (
                    <tr key={i}
                        className={
                          (i > 0 && f.label !== frames[i - 1].label ? "changed " : "") +
                          (f.low_confidence ? "low" : "")
                        }>
                      <td>{f.name}</td>
                      <td>{f.wetness_raw?.toFixed(1)}</td>
                      <td>{f.wetness?.toFixed(1)}</td>
                      <td>{f.slope?.toFixed(2)}</td>
                      <td>{f.state}</td>
                      <td>{f.trend}</td>
                      <td style={{ color: COLOR[f.label] }}>{f.label}</td>
                      <td>{f.confidence?.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}