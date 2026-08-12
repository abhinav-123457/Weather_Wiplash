"""
main.py - FastAPI entry point.

Both models load ONCE, in the lifespan handler, and stay resident. Measured
resident cost: ~1.1 GB for both.
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

from .config import (CLIP_WEIGHT, CONTEXT_DISAGREEMENT, CV_WEIGHT,
                     DAMP_THRESHOLD, DRY_REFERENCE_ANCHOR, DRY_THRESHOLD,
                     MIN_ROAD_COVERAGE, REFERENCE_OFFSET_LIMIT, ROI_PRESETS,
                     ROI_SQUARE_SIZE, ROI_TO_SQUARE, SEG_MODEL_ALT,
                     SHOT_CHANGE_SIM, USE_CONTEXT, USE_PHRASING,
                     USE_SEGMENTATION, WET_THRESHOLD)
from .models.segmentation import road_box_candidates
from .engine import weather as met
from .engine.suggestion import circuits, suggest
from .engine.temporal import SessionStore
from .models.clip_scorer import ClipScorer
from .models.cv_features import extract as extract_cv

ml: dict = {}
sessions = SessionStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    print("loading models...")

    # Off by default - see the evidence in config.USE_SEGMENTATION.
    if USE_SEGMENTATION:
        from .models.segmentation import RoadSegmenter
        ml["segmenter"] = RoadSegmenter()
        print(f"  SegFormer ready (road ids: {ml['segmenter'].road_ids})")
    else:
        ml["segmenter"] = None
        print("  segmentation disabled - ROI defines the track region")

    ml["clip"] = ClipScorer()
    print(f"  CLIP ready ({len(ml['clip'].prompts)} prompts cached)")

    if USE_PHRASING:
        from .models.phrasing import Phraser
        ml["phraser"] = Phraser()
    else:
        ml["phraser"] = None
    print(f"models loaded in {time.perf_counter() - t0:.1f}s")
    yield
    ml.clear()


app = FastAPI(title="Weather Whiplash", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------
def apply_roi(image: Image.Image, camera_type: str,
              roi: str | None) -> tuple[Image.Image, list[float]]:
    """Crop to the region that actually contains track surface.

    An onboard frame is mostly car: halo, mirrors, bodywork, front wheels.
    ADE20k has no class for any of that, so SegFormer scatters those pixels
    across whatever it finds nearest and road coverage collapses. Cropping to
    the upper band before segmenting removes the problem at source rather
    than trying to teach the model about racing cars.

    roi is "x0,y0,x1,y1" in 0-1 fractions; omit it to use the camera preset.
    """
    if roi:
        try:
            box = [float(v) for v in roi.split(",")]
            assert len(box) == 4
        except Exception:
            box = ROI_PRESETS.get(camera_type, [0, 0, 1, 1])
    else:
        box = ROI_PRESETS.get(camera_type, [0, 0, 1, 1])

    w, h = image.size
    x0, y0, x1, y1 = box
    crop = image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    return crop, box


def label_from(wetness: float) -> str:
    """State only. DRYING needs a trend - that arrives with the temporal layer."""
    if wetness < DRY_THRESHOLD:
        return "DRY"
    if wetness < DAMP_THRESHOLD:
        return "DAMP"
    return "WET"


def build_mask(cropped: Image.Image):
    """Decide which pixels count as track surface.

    With segmentation off, that is the entire ROI - the operator already
    said where the track is by choosing the crop.
    """
    if ml.get("segmenter") is None:
        h, w = cropped.height, cropped.width
        return np.ones((h, w), dtype=bool), 1.0, [], "roi"

    mask, coverage, top_classes, source = ml["segmenter"].mask(cropped)
    return mask, coverage, top_classes, source


def detect_track_roi(image: Image.Image) -> tuple[list | None, dict]:
    """Find the track surface: SegFormer proposes, CLIP disposes.

    Two models, two jobs, neither trusted alone:

      SegFormer (Cityscapes) is good at finding smooth road-like regions
      and bad at knowing what it found - measured live, its bottom-of-frame
      prior labelled an onboard car's dark bodywork "road", and the box
      landed on a Mercedes cockpit reading WET at confidence 0.72.

      CLIP cannot localise, but it can tell asphalt from a cockpit from TV
      graphics instantly. So every candidate box is cropped and checked
      before it is allowed anywhere near the wetness score.

    Returns (box | None, diagnosis). The diagnosis carries what was
    rejected and why, so a failure is explainable rather than mysterious.
    """
    seg = _auto_segmenter()
    if seg is None:
        return None, {"reason": "detector_unavailable"}

    mask, coverage, top, source = seg.mask(image)
    if source != "segformer" or coverage < MIN_ROAD_COVERAGE:
        return None, {"reason": "no_road_pixels", "top_classes": top}

    scorer = ml["clip"]
    rejected = []
    for cand in road_box_candidates(mask, k=3):
        w, h = image.size
        crop = image.crop((int(cand[0] * w), int(cand[1] * h),
                           int(cand[2] * w), int(cand[3] * h)))
        v = scorer.verify_crop(crop)
        if v["is_track"]:
            return cand, {"reason": "verified", "verify": v,
                          "coverage": round(coverage, 3),
                          "rejected": rejected}
        rejected.append({"box": [round(c, 3) for c in cand],
                         "looked_like": v["kind"],
                         "confidence": v["confidence"]})

    # Every candidate was bodywork, graphics or scenery. Saying so beats
    # scoring a cockpit.
    return None, {"reason": "candidates_rejected", "rejected": rejected,
                  "top_classes": top}


def analyse(image: Image.Image, camera_type: str, roi: str | None,
            s=None, auto_roi: bool = False) -> dict:
    scorer = ml["clip"]

    # ---- auto-ROI follow ----
    # Broadcast footage cuts between shots; a box placed for one shot is
    # wrong for the next. The full-scene embedding (needed by scene context
    # anyway) doubles as the cut detector: a big embedding jump = new shot,
    # and only then does the track detector re-place the box. Between cuts
    # the box is held, so the expensive segmenter runs a handful of times
    # per video, not per frame.
    scene_emb = None
    roi_source = "manual"
    if auto_roi and s is not None:
        scene_emb = scorer.embedding(image)
        cut = (s.last_scene_emb is None or
               float(np.dot(scene_emb, s.last_scene_emb)) < SHOT_CHANGE_SIM)
        s.last_scene_emb = scene_emb
        if cut or s.auto_roi_box is None:
            new_box, diag = detect_track_roi(image)
            if new_box is not None:
                s.auto_roi_box = new_box
                roi_source = "auto: shot change — track re-detected"
            else:
                # No verified track in this shot (cockpit close-up, crowd,
                # garage, podium). Fall back to the camera preset rather
                # than keep a box that belonged to different footage, and
                # name what was rejected so the screen explains itself.
                s.auto_roi_box = None
                rej = diag.get("rejected") or []
                what = rej[0]["looked_like"] if rej else diag.get("reason")
                roi_source = f"auto: no track ({what}) — preset fallback"
        else:
            roi_source = "auto: held"
        if s.auto_roi_box is not None:
            roi = ",".join(f"{v:.4f}" for v in s.auto_roi_box)

    cropped, box = apply_roi(image, camera_type, roi)
    rgb = np.array(cropped)

    mask, coverage, top_classes, mask_source = build_mask(cropped)
    masked = (cropped if ml.get("segmenter") is None
              else ml["segmenter"].apply(cropped, mask))

    # Square the crop so the WHOLE ROI reaches CLIP. Without this its
    # processor centre-crops a wide band down to its middle quarter - see
    # ROI_TO_SQUARE in config. Applied only to the CLIP input; the CV
    # features keep the true-aspect pixels, so mask and array stay aligned.
    clip_input = masked
    if ROI_TO_SQUARE:
        clip_input = masked.resize((ROI_SQUARE_SIZE, ROI_SQUARE_SIZE))

    road = None
    if scorer.probe is not None:
        # Probe gives the label directly from argmax - more accurate than
        # re-deriving it by thresholding the score it produced. The road
        # probe (second opinion, never votes) reads the SAME embedding, so
        # it costs nothing extra per frame.
        emb = scorer.embedding(clip_input)
        clip_wetness, probs, confidence, probe_label = scorer.probe_from(emb)
        road = scorer.road_opinion_from(emb)
        method = "probe"
    else:
        clip_wetness, probs, confidence = scorer.score(clip_input)
        probe_label = None
        method = "prompts"

    cv = extract_cv(rgb, mask, camera_type)
    wetness = CLIP_WEIGHT * clip_wetness + CV_WEIGHT * cv["combined"]
    state = probe_label.upper() if probe_label else label_from(wetness)

    # Scene context, read from the FULL frame rather than the ROI crop -
    # sky, shadows, crowd and spray are all outside the asphalt.
    # REPORTED, NOT FUSED: it has to earn a vote against labelled data first.
    # Reuses the auto-follow scene embedding when one was computed.
    context = scorer.score_context(image, scene_emb) if USE_CONTEXT else {}
    disagree = None
    if context:
        gap = abs(context["rain_likelihood"] - wetness)
        if gap > CONTEXT_DISAGREEMENT:
            disagree = (f"surface reads {wetness:.0f}/100 but the scene looks "
                        f"'{context['scene']}' ({context['rain_likelihood']:.0f}"
                        f"/100) - one of them is wrong")

    return {
        "state": state,
        "wetness_frame": round(wetness, 1),
        "confidence": round(confidence, 3),
        "method": method,
        "road_coverage": round(coverage, 4),
        "mask_source": mask_source,
        "roi": box,
        "roi_source": roi_source,
        "top_classes": top_classes,
        "context": context,
        "context_disagreement": disagree,
        # Second opinion from the public-road probe. Display only - the
        # decision above never sees it (measured night-race blind spot).
        "road_opinion": road,
        "signals": {
            "clip": round(clip_wetness, 1),
            "cv": cv,
            "clip_probabilities": [round(float(p), 3) for p in probs],
        },
    }



def add_phrasing(rec: dict) -> dict:
    """Let the model write the sentence, after the rules have decided it.

    Ordering matters: the decision exists before generation is attempted, so
    a failure here degrades to the deterministic template rather than
    producing no suggestion at all.
    """
    phraser = ml.get("phraser")
    if phraser is None:
        rec["message"] = (rec["headline"] if not rec.get("detail")
                          else f"{rec['headline']} — {rec['detail']}")
        rec["message_source"] = "template"
        return rec

    out = phraser.phrase(
        headline=rec["headline"], detail=rec.get("detail"),
        urgency=rec["urgency"], label=rec["basis"].split(" · ")[0],
        trend=rec["basis"].split(" · ")[-1], wetness=rec.get("_wetness", 0),
        current_tire=rec["current_tire"], suggested_tire=rec["suggested_tire"],
        laps=rec.get("laps_to_crossover"),
    )
    rec["message"] = out["text"]
    rec["message_source"] = out["source"]
    if "rejected" in out:
        rec["message_rejected"] = out["rejected"]
    return rec


# --------------------------------------------------------------------------
@app.get("/health")
def health():
    seg = ml.get("segmenter")
    scorer = ml.get("clip")
    probe = getattr(scorer, "probe", None) if scorer else None
    return {
        "status": "ok",
        "models_loaded": scorer is not None,
        # segmentation is off by default, so this is normally None - guard on
        # the object, not on the dict being non-empty.
        "segmentation": ("enabled" if seg else "disabled"),
        "road_class_ids": seg.road_ids if seg else None,
        "scoring_method": "probe" if probe else "prompts",
        "probe_cv_accuracy": (round(probe["cv_accuracy"], 3)
                              if probe else None),
        "active_sessions": len(sessions._s),
    }


@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    camera_type: str = Form("trackside"),
    session_id: str = Form("default"),
    roi: str | None = Form(None),
    auto_roi: bool = Form(False),
    # ---- strategy inputs ----
    # None of this is inferable from a photograph. A camera cannot see tire
    # age, lap count or humidity - but a real pit wall already has all of it.
    circuit: str = Form("silverstone"),
    current_tire: str = Form("INTER"),
    # Left as None so the circuit's typical race-day conditions are used
    # unless the operator overrides them. A fixed global default made every
    # circuit dry at the same rate.
    track_temp: float | None = Form(None),
    air_temp: float | None = Form(None),
    humidity: float | None = Form(None),
    wind_speed: float | None = Form(None),
):
    t0 = time.perf_counter()
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    s = sessions.get(session_id)
    frame = analyse(image, camera_type, roi, s=s, auto_roi=auto_roi)

    if "error" in frame:
        frame["session_id"] = session_id
        return frame

    # Dry-reference calibration. The classifier's raw reading is kept BEFORE
    # any adjustment, so /api/reference always calibrates from what the model
    # actually said - setting the reference twice must not compound.
    s.last_uncalibrated = frame["wetness_frame"]
    if s.reference_offset is not None:
        adjusted = min(100.0, max(0.0, frame["wetness_frame"]
                                  - s.reference_offset))
        frame["wetness_uncalibrated"] = frame["wetness_frame"]
        frame["wetness_frame"] = round(adjusted, 1)
        # With an operator-supplied anchor, thresholds on the adjusted score
        # outrank the probe's argmax - the probe's absolute calibration is
        # exactly what was just corrected.
        frame["state"] = label_from(adjusted)
        frame["reference_offset"] = round(s.reference_offset, 1)

    # Feed the frame into this session's history. The temporal layer owns
    # the final label - it is the only part that can see direction.
    temporal = s.add(
        frame["wetness_frame"], frame["state"], frame["confidence"])
    frame.update(temporal)

    # Strategy runs on the SMOOTHED wetness and the COMMITTED label, never on
    # the raw frame. A tire call should not swing on one noisy image.
    frame["recommendation"] = add_phrasing(suggest(
        label=frame["label"], trend=frame["trend"],
        wetness=frame["wetness"], confidence=frame["confidence"],
        circuit_key=circuit, current_tire=current_tire.upper(),
        track_temp=track_temp, air_temp=air_temp,
        humidity=humidity, wind_speed=wind_speed,
    ))

    frame["session_id"] = session_id
    frame["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
    return frame


@app.post("/api/strategy")
def strategy_only(
    session_id: str = Form("default"),
    circuit: str = Form("silverstone"),
    current_tire: str = Form("INTER"),
    track_temp: float | None = Form(None),
    air_temp: float | None = Form(None),
    humidity: float | None = Form(None),
    wind_speed: float | None = Form(None),
):
    """Recompute the tyre call from the session's CURRENT condition.

    No image required. Changing tyre age or lap count does not change what
    the track looks like, so re-uploading a frame to see the effect is both
    wasteful and confusing - the operator expects the call to update as they
    adjust the inputs, the way a real strategy tool does.
    """
    s = sessions.get(session_id)
    if not s.smooth:
        return {"error": "no_frames",
                "message": "Analyse at least one frame first."}

    return {
        "session_id": session_id,
        "based_on": {
            "label": s.labels[-1], "wetness": round(s.smooth[-1], 1),
            "frames": len(s.smooth),
        },
        "recommendation": add_phrasing(suggest(
            label=s.labels[-1], trend=s.last_trend, wetness=s.smooth[-1],
            confidence=s.last_confidence, circuit_key=circuit,
            current_tire=current_tire.upper(),
            track_temp=track_temp, air_temp=air_temp,
            humidity=humidity, wind_speed=wind_speed,
        )),
    }


@app.post("/api/reference")
def set_reference(session_id: str = Form("default"),
                  clear: bool = Form(False)):
    """Mark the session's LAST analysed frame as known-dry track.

    Deployment flow: point the camera at track the operator KNOWS is dry,
    analyse one frame, mark it. The gap between that reading and what a
    well-calibrated dry track reads becomes a constant subtracted from every
    later frame - this is the per-camera answer to the measured 16-28 point
    venue offset that no global threshold survives.

    Mark the reference EARLY: frames analysed before it keep their
    uncalibrated scores, so a mid-session reference puts a visible step in
    the chart.
    """
    s = sessions.get(session_id)
    if clear:
        s.reference_offset = None
        return {"status": "cleared", "session_id": session_id}

    if s.last_uncalibrated is None:
        return {"error": "no_frames",
                "message": "Analyse a frame showing dry track first."}

    offset = s.last_uncalibrated - DRY_REFERENCE_ANCHOR
    if abs(offset) > REFERENCE_OFFSET_LIMIT:
        # A "dry" frame reading 70 was not dry. Refusing beats silently
        # mis-calibrating every subsequent frame in the session.
        return {"error": "implausible_reference",
                "message": (f"That frame read {s.last_uncalibrated:.1f} - "
                            f"more than {REFERENCE_OFFSET_LIMIT:.0f} from a "
                            f"dry reading. If the track in it is truly dry, "
                            f"check the ROI placement instead.")}

    s.reference_offset = offset
    return {"status": "set", "session_id": session_id,
            "offset": round(offset, 1),
            "based_on_wetness": round(s.last_uncalibrated, 1)}


@app.get("/api/weather/{circuit_key}")
def circuit_weather(circuit_key: str):
    """Live conditions at the circuit, for display in the UI.

    Source is always explicit: 'live' means Open-Meteo answered for the
    circuit's coordinates just now; 'typical' means the feed is unreachable
    and the UI must say so instead of showing a guess as a measurement.
    """
    c = circuits().get(circuit_key)
    if not c:
        return {"error": "unknown_circuit", "circuit": circuit_key}
    lw = met.live(c.get("lat"), c.get("lon"))
    if lw is None:
        return {"circuit": circuit_key, "source": "typical",
                "weather": c.get("typical_weather", {})}
    return {"circuit": circuit_key, "source": "live", "weather": lw}


@app.get("/api/circuits")
def list_circuits():
    """Circuit profiles, for populating the UI dropdown."""
    return {
        k: {"name": v.get("name", k), "country": v.get("country"),
            "severity": v.get("severity"),
            "surface_type": v.get("surface_type"), "shade": v.get("shade"),
            "lap_time_s": v.get("lap_time_s"), "pit_loss_s": v.get("pit_loss_s"),
            # Sent so the UI can show each circuit's typical conditions as
            # placeholders - the operator sees what will be used if they
            # leave a field blank.
            "typical_weather": v.get("typical_weather", {})}
        for k, v in circuits().items() if not k.startswith("_")
    }


@app.post("/api/session/reset")
def reset_session(session_id: str = Form("default")):
    """Clear a session's history.

    Essential before a second demo run: without it the new frames append to
    the old ones and the trend line becomes meaningless.
    """
    sessions.reset(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    s = sessions.get(session_id)
    return {
        "session_id": session_id,
        "frames": len(s.raw),
        "labels": s.labels,
        "states": s.states,
        "chart_data": [
            {"frame": i + 1, "raw": round(r, 1), "smooth": round(sm, 1)}
            for i, (r, sm) in enumerate(zip(s.raw, s.smooth))
        ],
    }


def _auto_segmenter():
    """Cityscapes SegFormer for one-shot track detection, loaded lazily.

    Per-frame segmentation stays off: ADE20k measured 0% road on real F1
    frames. This is a different job with different odds - the CITYSCAPES
    checkpoint is trained on road-facing dashcam footage (much closer to
    onboard F1), and it only has to place a BOX once per camera, not
    survive every frame. Lazy so startup pays nothing for the common path
    that never clicks the button.
    """
    if "auto_seg" not in ml:
        try:
            from .models.segmentation import RoadSegmenter
            ml["auto_seg"] = RoadSegmenter(model_name=SEG_MODEL_ALT)
            print(f"  auto-ROI segmenter ready ({SEG_MODEL_ALT})")
        except Exception as exc:
            print(f"  auto-ROI segmenter unavailable ({type(exc).__name__})")
            ml["auto_seg"] = None
    return ml["auto_seg"]


@app.post("/api/roi/suggest")
async def suggest_roi(file: UploadFile = File(...)):
    """Detect the track in this frame and suggest an ROI box.

    A SUGGESTION, not an oracle: the returned box is placed in the same
    draggable control the operator already owns, so a bad detection costs
    one drag rather than a corrupted session. When no track is found the
    endpoint says so - with what the model saw instead - rather than
    returning a full-frame box that would silently score sky and barriers.
    """
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    box, diag = detect_track_roi(image)

    if box is None:
        if diag["reason"] == "detector_unavailable":
            msg = "Track detector could not load — draw the ROI by hand."
        elif diag["reason"] == "candidates_rejected":
            # Name what it actually found. "Looked like a cockpit" tells the
            # operator to point at track; a bare failure tells them nothing.
            rej = diag["rejected"][0]
            msg = (f"No track found — the best region looked like "
                   f"{rej['looked_like']}. Draw the ROI by hand.")
        else:
            seen = ", ".join(f"{t['label']} {t['pct']}%"
                             for t in diag.get("top_classes", [])[:3])
            msg = (f"No track surface detected — draw the ROI by hand."
                   + (f" The model saw: {seen}" if seen else ""))
        return {"error": "no_track_found", "message": msg, "diagnosis": diag}

    return {"roi": [round(v, 4) for v in box],
            "coverage": diag.get("coverage"),
            "verified_as": diag["verify"],
            "rejected": diag.get("rejected"),
            "model": f"{SEG_MODEL_ALT} + CLIP verification"}


@app.post("/api/debug/mask")
async def debug_mask(
    file: UploadFile = File(...),
    camera_type: str = Form("trackside"),
    roi: str | None = Form(None),
):
    """Return the analysed region with non-road pixels dimmed.

    Look at this FIRST whenever a score seems wrong. Every downstream number
    is computed from these pixels, so if the mask is bad the score cannot be
    right - and no amount of tuning thresholds will fix it.
    """
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    cropped, _ = apply_roi(image, camera_type, roi)
    mask, coverage, top_classes, mask_source = build_mask(cropped)

    arr = np.array(cropped).astype(np.float32)
    arr[~mask] *= 0.25                       # keep context, but clearly dimmed
    arr[mask] = np.clip(arr[mask] * 1.15, 0, 255)
    out = Image.fromarray(arr.astype(np.uint8))

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Road-Coverage": f"{coverage:.4f}",
            "X-Mask-Source": mask_source,
            # Headers, not the body, so the PNG stays a plain image while the
            # diagnosis travels with it.
            "X-Top-Classes": ", ".join(
                f"{c['label']}:{c['pct']}%" for c in top_classes),
        },
    )