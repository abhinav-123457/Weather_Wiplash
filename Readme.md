# Weather Whiplash — Live Track Condition Detector

Reads camera frames of a racetrack, decides whether the surface is **Dry,
Damp, Wet or Drying**, tracks how it changes over time, and turns that into a
tyre suggestion.

Built for Problem Statement 02. React frontend, FastAPI backend, Hugging Face
models, CPU-only, runs offline after the first launch.

```
  IMAGE / VIDEO FRAME / WEBCAM
        |
   [1] ROI crop              operator marks the track surface
        |
   [2] CLIP ViT-B/32         512-d embedding            (Hugging Face Hub)
        |
   [3] Linear probe          dry / damp / wet + confidence
        |
   [4] Temporal layer        EMA -> slope -> trend -> the four labels
        |                    asymmetric hysteresis, confidence gating
   [5] Suggestion rules      tyre safety bands, urgency
        |
   React dashboard           label + trend graph + suggestion + why
```

---

## Table of contents

- [The idea](#the-idea)
- [Quick start](#quick-start)
- [How to use it](#how-to-use-it)
- [How it works, in depth](#how-it-works-in-depth)
- [Results, honestly](#results-honestly)
- [What was tried and removed](#what-was-tried-and-removed)
- [Repository layout](#repository-layout)
- [API reference](#api-reference)
- [Retraining the classifier](#retraining-the-classifier)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## The idea

The brief asks for four labels: Dry, Damp, Wet, Drying.

**Three describe how much water is on the track. One describes which way it is
moving** — and a single frozen frame cannot answer the fourth. Identical pixels
mean opposite things depending on what came before:

```
wetness 45, arrived from 80   ->   DRYING    slick window opening
wetness 45, arrived from 15   ->   WETTING   get intermediates on
```

Same frame, same score, opposite tyre call. So the system splits the problem: a
classifier answers *how wet* from one image, a temporal layer answers *which
direction* from the sequence, and the two combine into the four required labels.

Independent confirmation that this is the right split: the **RoadSaW** dataset
(CVPR-W 2022), built with a calibrated water-film sensor, labels *dry / damp /
wet / very wet* — all states, no "drying". Nobody labels drying from a single
image, because it is not in the image.

---

## Quick start

**Requirements:** Python 3.12+, Node 20+, ~2 GB free disk (model weights),
internet on first run only.

### 1. Clone

```bash
git clone https://github.com/abhinav-123457/Weather_Wiplash.git
cd Weather_Wiplash
```

### 2. Backend

<details open>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=5.0,<6.0" pillow numpy opencv-python-headless `
            fastapi "uvicorn[standard]" python-multipart huggingface_hub `
            scikit-learn psutil

python -m uvicorn backend.app.main:app --port 8000
```

</details>

<details>
<summary><b>Linux / macOS</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=5.0,<6.0" pillow numpy opencv-python-headless \
            fastapi "uvicorn[standard]" python-multipart huggingface_hub \
            scikit-learn psutil

python -m uvicorn backend.app.main:app --port 8000
```

</details>

Wait for:

```
loading models...
  segmentation disabled - ROI defines the track region
  probe loaded: 83.3% cross-validated
  CLIP ready (13 prompts cached)
models loaded in 6.2s
```

The **first** launch downloads CLIP (~600 MB) from the Hugging Face Hub. Every
launch after that is offline and takes a few seconds.

> `probe loaded` is the line that matters. If it says *"probe.npz is missing —
> falling back to prompts"*, accuracy drops from 83% to 54%. See
> [Troubleshooting](#troubleshooting).

### 3. Frontend

In a **second terminal**:

```bash
cd frontend
npm install
npm run dev
```

Open **<http://localhost:5173>**.

Vite proxies `/api` and `/health` to port 8000, so the browser sees a single
origin and CORS never enters the picture.

### 4. Check it works

```bash
python setup/smoke_test.py
```

Verifies imports, loads the models, and reports measured per-frame timing.

---

## How to use it

The app needs **frames of track surface** — screenshots from race footage, a
video file, or a live camera.

### Step 1 — pick a source

| Mode | Use it for |
|---|---|
| **Still frames** | A folder of screenshots. Runs in filename order, so name them `race_01.png`, `race_02.png`… |
| **Video file** | Any local video. Samples a frame every *N* seconds while it plays. Pause the video and sampling pauses too — useful for skipping broadcast cut-aways. |
| **Camera** | A webcam or phone camera. The real deployment shape. |

Also pick **Trackside view** or **Onboard view** — this sets the default ROI.

### Step 2 — check the ROI (the most important control)

The white box is **exactly what gets scored**. Everything inside it is treated
as track surface, so a box containing sky, barriers or bodywork produces a
meaningless number.

- **Drag on the image** to redraw it.
- The pill shows `scoring 34% of frame` — how much is actually being read.
- **Test ROI** scores the current frame in a throwaway session, so you can tune
  the box without polluting the trend line.

### Step 3 — analyse

The right-hand column fills in with the three outputs the brief asks for:

- the **label** (Dry / Damp / Wet / Drying) with a wetness score and trend
- the **suggestion** — a message and a tyre call
- the **trend graph**, with threshold bands so the number has meaning

Below that, **Why this call** shows every check the backend ran — including the
ones that failed, because knowing why the system is *not* calling a trend is as
useful as knowing why it is.

### Step 4 (recommended) — set a dry reference

Absolute scores shift between venues and cameras by 16–28 points (measured).
The fix is one click:

1. Analyse a frame you **know** shows dry track.
2. Click **mark last frame as known dry**.

The offset is subtracted from every later frame in the session. If the frame was
not actually dry the backend refuses it rather than mis-calibrating silently.

---

## How it works, in depth

### [1] ROI — the operator defines the track

Segmentation is **disabled by default** (see
[what was removed](#what-was-tried-and-removed)), so the ROI *is* the mask.

One non-obvious detail matters a lot: `CLIPProcessor` resizes the **shortest**
side to 224 and then centre-crops. A wide flat band therefore loses most of its
width before the model sees it:

| preset | drawn | aspect | width CLIP kept |
|---|---|---|---|
| onboard | 1613×443 | 3.6:1 | ~25% |
| trackside | 1805×756 | 2.4:1 | ~40% |

Three-quarters of an onboard selection was being discarded — which also
explained why the score was so sensitive to exactly where the box sat.
`ROI_TO_SQUARE = True` squares the crop first, so the **whole** selection
reaches CLIP. Effective coverage went from 7.7% of the frame to 34.4%.

### [2] CLIP — a frozen feature extractor

`openai/clip-vit-base-patch32`, 88M parameters, **never fine-tuned**. It turns
an image into a 512-d vector. Prompts are kept as a fallback and as the thing
the probe is measured against.

> **transformers v5 note.** v4 returned a plain tensor from `get_*_features()`;
> v5 returns `BaseModelOutputWithPooling` whose `pooler_output` is *already*
> projected. Projecting again fails loudly for vision but **succeeds silently**
> for text on ViT-B/32 (512→512), producing wrong embeddings with a
> correct-looking shape. `_as_embedding()` compares against
> `config.projection_dim` to catch exactly that.

### [3] Linear probe — ~1,536 numbers

A logistic regression over the frozen embeddings, in `backend/app/probe.npz`.
Not fine-tuning: CLIP's weights are untouched, and this trains on CPU in
seconds from ~100 labelled frames.

The **label** comes from `argmax`, not from thresholding — the probe already
outputs class probabilities. The **continuous wetness** is still needed because
the temporal layer works on slope, and a categorical label has no slope.

### [4] Temporal layer — where "drying" comes from

`backend/app/engine/temporal.py`. Per session:

1. **EMA smoothing** (`α = 0.45`) — raw per-frame scores are too noisy for a
   usable slope.
2. **Least-squares slope** over the last 5 frames. Fitted, not
   `last − first`: endpoint differencing lets one noisy frame dictate the
   trend. The **full** window is required — a partial fit measures the EMA
   settling, which once produced a phantom DRYING on twelve soaking frames.
3. **Trend** — `DRYING` if slope ≤ −2.5 **and** wetness is above 45 (a small
   negative slope on a dry track is noise, not drying).
4. **The four labels** — state gives three, direction gives the fourth.
5. **Asymmetric hysteresis** — improving conditions need **3** consecutive
   frames, worsening only **2**. Slicks too early is a crash; slicks too late
   costs a few seconds a lap. Direction is judged by `LABEL_SEVERITY`, not
   string equality, so a track escalating DRY→DAMP→WET still accumulates
   evidence instead of resetting at every step.
6. **Confidence gating** — below `CONFIDENCE_MIN = 0.50` the frame still nudges
   the score at half weight, but is barred from moving the label.

### [5] Suggestion — deterministic, and deliberately small

`backend/app/engine/suggestion.py`. 12 headlines keyed by `(label, trend)`,
plus a safety check against `TIRE_BAND` — the wetness range each tyre can
actually be used in.

Safety is never traded against lap time. An earlier version had no branch for
"slicks on a wet track", so it fell through to pit-loss economics and advised
staying out — on a car with no tread in standing water. Stating a safe band for
every tyre makes that class of gap impossible.

**Live weather** comes from Open-Meteo (free, no API key) using each circuit's
coordinates, and every value on screen is labelled with its source —
`live`, `operator`, or `typical`. If the feed is unreachable the UI says so
rather than showing a guess as a measurement. Rain at the circuit appears as a
note; it is **reported, never fused** into the label.

---

## Results, honestly

| | |
|---|---|
| **Wet vs not-wet, venue-held-out** | **85.4%** |
| 3-class, venue-held-out (LORO) | 36.5% |
| Damp < wet ordering within a venue | Monaco, **2.01 pooled SD** |
| Trend layer, known-shape sequences | **7 / 7** |
| Per frame, CPU (i7-1355U) | **~0.11 s** |

> **On the 83.3% printed at startup:** that is *random* 5-fold cross-validation
> over 96 images drawn from race sessions. Frames from the same race are
> near-duplicates, so a random split trains on siblings of its own test data.
> Re-validating with **entire races held out** dropped 3-class accuracy to
> 36.5%. The honest headline is **85.4% wet-vs-not-wet, venue-held-out**.

### The scale-up experiment that failed, and why that is useful

We retrained on ~1M public road images (**RSCD** + **RoadSaW**) with zero F1
frames, then judged the result against our F1 set with three acceptance gates.

| probe | 3-class | dry frames kept dry | Monaco separation |
|---|---|---|---|
| F1-only (shipped) | 36.5% | — | 2.01 SD |
| Road-only (1M images) | **49.0%** | **38.7%** ✗ | 0.63 SD ✗ |
| Road + F1 mixed (10/30/50%) | 26–30% ✗ | 6–16% ✗ | ~1.5 SD ✗ |

Road-only scored *higher overall* but failed the tyre-mark gate: **every miss
was Singapore**, the night race. RSCD and RoadSaW are daytime datasets, so a
model that has never seen floodlights reads glare on dry asphalt as standing
water — 94 to 100 out of 100, confidently.

Mixing was worse than either endpoint, in two independent runs. A single
1,536-parameter hyperplane cannot hold *"glare means wet"* for daytime roads
and *"glare means dry floodlights"* for night races at the same time.

So the road model ships as a **second opinion that never votes** — displayed in
the Why panel beside the decision. Drop a `probe_clip_both.npz` next to
`probe.npz` and the row appears; the file's absence turns the feature off.

---

## What was tried and removed

Seven features, each physically plausible, each killed by measurement rather
than argument. This is the part of the project we would most want read.

| Feature | Why it went |
|---|---|
| **SegFormer segmentation** | 290 ms/frame — 73% of the pipeline — and **0% road found** on four real broadcast frames. It reads circuit barriers as buildings. |
| **Darkness** | Scored a bone-dry photo of fresh dark tarmac at **84.5/100** wetness. Asphalt colour varies with age and rubber; brightness carries almost no wetness signal. |
| **Texture (Laplacian)** | Two equally wet Monaco frames gave variances of **25.9 and 487.1** — a 19× swing — because one was motion-blurred. It measured camera sharpness. |
| **Specular reflection** | **17.5 / 1.0 / 23.3** across three equally wet frames. Broadcast auto-exposure keeps wet road below any highlight threshold. |
| **Spatial sub-bands** | Band median scored **worse** than the full ROI (1.81 vs 2.01 SD); monotonicity sat at chance and was identical across classes. |
| **Auto-ROI detection** | Three attempts. Cityscapes' bottom-of-frame prior labels dark bodywork "road", so the box landed on a car's engine cover and read **WET 75**. Kept as an experimental toggle for fixed cameras; the preset plus one drag wins on cut broadcast footage. |
| **flan-t5 phrasing** | Generated *"The weather is a bit dry and wet"* three times in one message. Guards now catch degenerate repetition, but `USE_PHRASING = False` — the deterministic templates read better and every word can be defended. |

We also retired the claim of "three independent signals": darkness and the CLIP
"damp" prompt both keyed on brightness, so they failed *together* rather than
differently — the opposite of independence.

The classical CV features still compute and are reported as diagnostics. They
simply do not vote.

---

## Repository layout

```
backend/app/
  config.py              every tunable, with the measurement behind it
  main.py                FastAPI app, routes, model lifespan
  probe.npz              the trained classifier (committed — 1,536 numbers)
  models/
    clip_scorer.py       CLIP embedding, prompt ensemble, probe, scene context
    cv_features.py       classical CV signals (reported, not scored)
    phrasing.py          flan-t5 message writer with guards (off by default)
    segmentation.py      SegFormer wrapper — disabled, kept as the record
  engine/
    temporal.py          EMA, slope, hysteresis, confidence gating, evidence
    suggestion.py        tyre safety bands, urgency, headlines
    weather.py           Open-Meteo live conditions, cached, honest fallback

frontend/src/
  App.jsx                layout, capture modes, sessions, draggable ROI
  TrendChart.jsx         hand-drawn SVG chart with threshold bands
  StrategyPanel.jsx      the suggestion message
  WhyPanel.jsx           the reasoning behind the current call
  RaceInputs.jsx         circuit, tyre, live weather readout
  styles.css             F1 broadcast styling (flag semantics)

tools/
  train_probe.py         fit the linear probe on labelled images
  train_probe_v2.ipynb   Colab: retrain on RSCD/RoadSaW + acceptance gates
  validate_loro.py       leave-one-race-out validation
  calibrate.py           threshold + temperature sweep
  inspect_scores.py      per-venue score distributions
  inspect_bands.py       spatial sub-band experiment
  test_trend.py          temporal layer, known-shape sequences
  run_sequence.py        push a folder of frames through the API
  make_test_sequence.py  build an ordered sequence from calibration frames
  refresh_circuits.py    measure lap times / pit losses from real sessions

circuits.json            24 circuits, every field marked with its provenance
setup/smoke_test.py      environment check with measured timing
SYSTEM.md                full technical documentation
docs/BUILD_PLAN.md       the original plan (superseded — see note below)
```

---

## API reference

Base URL `http://127.0.0.1:8000`. Interactive docs at `/docs`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | model status, scoring method, active sessions |
| `POST` | `/api/analyze/image` | **the main endpoint** — one frame in, label + trend + suggestion out |
| `POST` | `/api/strategy` | recompute the tyre call without re-uploading a frame |
| `POST` | `/api/reference` | mark the last frame as known-dry (per-camera calibration) |
| `GET` | `/api/weather/{circuit}` | live conditions at the circuit |
| `GET` | `/api/circuits` | circuit list for the dropdown |
| `POST` | `/api/session/reset` | clear a session's history |
| `GET` | `/api/session/{id}` | full frame history and chart data |
| `POST` | `/api/roi/suggest` | experimental track detection |
| `POST` | `/api/debug/mask` | returns the analysed region as a PNG — **look at this first when a score seems wrong** |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/image \
  -F "file=@frame.jpg" \
  -F "camera_type=trackside" \
  -F "session_id=demo" \
  -F "roi=0.03,0.20,0.97,0.90" \
  -F "circuit=silverstone" \
  -F "current_tire=INTER"
```

Returns `label`, `state`, `trend`, `wetness`, `wetness_raw`, `slope`,
`confidence`, `evidence[]`, `chart_data[]` and `recommendation{}`.

---

## Retraining the classifier

`backend/app/probe.npz` is committed, so **the repo works as cloned**. Retrain
only if you have your own labelled frames.

### On your own data

```
calibrate/
  dry/    monaco2024_trackside_01.png  ...
  damp/   monaco2023_onboard_03.png    ...
  wet/    monaco2023_onboard_07.png    ...
```

The filename prefix before the first underscore is treated as the **race
identity** — that is what leave-one-race-out holds out, so name files
consistently or the validation number becomes meaningless.

```bash
python tools/train_probe.py        # writes backend/app/probe.npz
python tools/validate_loro.py      # the honest, venue-held-out number
```

Restart the backend. Startup prints the new accuracy.

### On public road datasets (Colab)

`tools/train_probe_v2.ipynb` is a self-contained pipeline: downloads RSCD and
RoadSaW, extracts a balanced subset, embeds with CLIP **and** DINOv2, trains
probes, and runs the three acceptance gates against your F1 frames. Read
[the results section](#results-honestly) first — road-only training fails the
night-race gate, and that notebook is how we found out.

---

## Known limitations

**Night races are the weak spot**, measured four separate ways. Floodlights
create real specular reflections on *dry* asphalt, so a dry night track and a
wet one look far more alike than their daytime equivalents. The system degrades
honestly: confidence drops below threshold and the label is held rather than
guessed.

**Dry versus damp is unestablished across venues.** No venue in the dataset
contains both, so the model could only learn venue identity. The most valuable
data that could be collected is **one venue in both dry and damp conditions**.

**Absolute scores do not transfer between circuits.** The per-venue offset is
16–28 points, comparable to the between-class gap. The deployment answer is the
dry-reference button, not a global threshold.

**The 96-frame training set is a prototype dataset.** Real deployment means a
fixed camera per venue plus a few hundred labelled frames from *that* camera —
within one fixed view the confounds above largely vanish, and the signal is
demonstrably there (Monaco damp→wet separates at 2.01 SD).

**The wetness scale is not calibrated to a physical quantity.** It is 0–100,
not millimetres of water film. RoadSaW ships MARWIS sensor ground truth, which
is the path to fixing that.

**Real-world temporal transitions are not validated.** The trend layer passes
7/7 on known-shape sequences, but we had no continuous wet-to-dry footage to
test against.

**No dry-line detection.** The racing line dries first, and that *is* the real
F1 decision — but the spatial-band experiment showed no usable signal, so we
did not build a feature we could not defend.

---

## Troubleshooting

**`probe.npz is missing — falling back to prompts`**
The trained classifier is not where the backend expects it. Confirm
`backend/app/probe.npz` exists (it is committed — `git checkout --
backend/app/probe.npz` restores it). Prompt fallback measures 54% vs 83%.

**Everything reads WET, or everything reads DRY**
Look at the ROI first. Open `/api/debug/mask` with your frame, or just check
the white box on screen — if it covers sky, barriers or car bodywork, the score
is meaningless. Then set a dry reference.

**Frontend loads but shows "backend offline"**
The backend is still loading models (first run downloads ~600 MB), or it is not
on port 8000. Check the uvicorn terminal.

**`ImportError` on `transformers`**
Requires v5. `pip install "transformers>=5.0,<6.0"`.

**Video mode samples nothing**
Sampling is skipped while the video is paused — by design, so scrubbing does
not stack identical frames. Press play.

**Scores changed after an update**
`ROI_TO_SQUARE` changes what CLIP sees. Set it to `False` in `config.py` to
restore the previous behaviour exactly.

---

## Notes

`docs/BUILD_PLAN.md` describes the **original** architecture — SegFormer
segmentation and a full race-strategy engine. Both were removed on evidence. It
is kept as the record of what was planned before the measurements changed it,
and it contradicts the current system by design. **`SYSTEM.md` is
authoritative.**

The `calibrate/` directory holds labelled training frames captured from race
broadcasts. It is **excluded from version control** — those frames are
third-party broadcast content and are not ours to redistribute. `probe.npz`,
the classifier derived from them, *is* committed, because it is 1,536 numbers
rather than imagery.

Dataset licences: **RSCD** is CC BY-NC, **RoadSaW** is CC BY-NC-SA. Fine for
research and this competition; a commercial deployment would need a probe
trained on licensed or self-collected data.

---

## Against the brief

- **Frontend and backend** — React dashboard and FastAPI service over HTTP, not a notebook
- **Uses the Hugging Face Hub** — CLIP for vision, flan-t5-small available for language, both local
- **All four labels** — Dry, Damp and Wet from the classifier, Drying from the temporal layer
- **Trend graph** — raw and smoothed series with threshold bands and label-change markers
- **Suggestion message** — deterministic, safety-first, with the reasoning shown
- **Balanced difficulty** — a classifier trained on our own labelled data over a frozen foundation model, a validated temporal layer, and a deterministic decision layer. Not one ready-made call; nothing from scratch.