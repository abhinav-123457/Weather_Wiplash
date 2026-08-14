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
        |                    optionally LoRA-adapted on our own frames
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
- [Results](#results)
- [The confound, and how it was found](#the-confound-and-how-it-was-found)
- [What was tried and removed](#what-was-tried-and-removed)
- [Repository layout](#repository-layout)
- [Building a dataset](#building-a-dataset)
- [API reference](#api-reference)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Licence](#licence)

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

**Requirements:** Python 3.12+, Node 20+, ~2 GB free disk, internet on first
run only.

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
pip install "transformers>=5.0,<6.0" peft pillow numpy opencv-python-headless `
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
pip install "transformers>=5.0,<6.0" peft pillow numpy opencv-python-headless \
            fastapi "uvicorn[standard]" python-multipart huggingface_hub \
            scikit-learn psutil

python -m uvicorn backend.app.main:app --port 8000
```

</details>

Wait for:

```
loading models...
  segmentation disabled - ROI defines the track region
  CLAHE preprocessing: off
  LoRA: building vision tower from CLIPVisionConfig (hidden_size=768)
  LoRA vision tower: ON  (adapter alters embeddings by 2.395 max)
  probe loaded: 69.3% cross-validated
  CLIP ready (13 prompts cached)
```

The **first** launch downloads CLIP (~600 MB). Every launch after is offline.

> Every line that changes what the model *sees* prints at startup. If `LoRA
> vision tower` says anything other than `ON`, it fell back to frozen CLIP and
> named the reason — a perception layer that fails silently is worse than one
> that fails.

### 3. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **<http://localhost:5173>**.

### 4. Check it works

```bash
python setup/smoke_test.py       # imports, models, per-frame timing
python tools/verify_lora.py      # is the adapter actually live?
```

---

## How to use it

### Step 1 — pick a source

| Mode | Use it for |
|---|---|
| **Still frames** | A folder of screenshots, run in filename order |
| **Video file** | Samples a frame every *N* seconds while it plays. Pause the video and sampling pauses too — useful for skipping broadcast cut-aways |
| **Camera** | A webcam or phone. The real deployment shape |

Also pick **Trackside** or **Onboard** — this sets the default ROI.

### Step 2 — check the ROI

The white box is **exactly what gets scored**. Drag on the image to redraw it;
the pill shows what fraction of the frame is being read.

The presets are deliberately large (trackside covers the whole frame) because
the classifier is *trained* on whole frames — see
[How it works](#1-roi--matching-what-the-model-was-trained-on).

### Step 3 — analyse

The right-hand column fills in with the three outputs the brief names: the
**label**, the **suggestion**, and the **trend graph** with threshold bands so
the wetness number has meaning.

Below that, **Why this call** shows every check the backend ran — including the
ones that failed, because knowing why the system is *not* calling a trend is as
useful as knowing why it is.

### Step 4 (recommended) — set a dry reference

Absolute scores shift between venues and cameras. Analyse a frame you **know**
shows dry track, click **mark last frame as known dry**, and the offset is
removed for the rest of the session. If the frame was not actually dry, the
backend refuses rather than mis-calibrating silently.

---

## How it works, in depth

### [1] ROI — matching what the model was trained on

Segmentation is disabled (see [what was removed](#what-was-tried-and-removed)),
so the ROI *is* the mask.

The presets are wide on purpose. `tools/extract_frames.py` writes **whole
frames** and `train_probe.py` embeds them whole, so the classifier has only
ever seen complete broadcast pictures. Serving it a tight strip of asphalt
hands it a kind of image it was never fitted on — a train/serve mismatch that
existed from day one and was invisible because both halves looked reasonable
alone.

```python
ROI_PRESETS = {"onboard":   [0.03, 0.20, 0.97, 0.90],   #  66% of frame
               "trackside": [0.00, 0.00, 1.00, 1.00]}   # 100%
ROI_TO_SQUARE = False    # CLIPProcessor's own resize+crop, as in training
```

### [2] CLIP — frozen, or LoRA-adapted

`openai/clip-vit-base-patch32`. Two modes, one flag:

- **`USE_LORA = False`** — frozen encoder, 512-d embedding. The original design.
- **`USE_LORA = True`** — the same tower with LoRA adapters on the attention
  projections, fine-tuned on our own labelled frames. ~591k trainable
  parameters, about 1% of the model.

The adapted tower is loaded **separately** from the `CLIPModel` used for text.
Both encoders contain modules named `q_proj`/`k_proj`/`v_proj`/`out_proj`, so
adapting the whole model would silently put untrained adapters on the text
tower and change every prompt in the system.

For the same reason, `embedding()` (probe path) and `embedding_frozen()` (all
text comparisons) are separate methods. CLIP's image and text spaces are
aligned by joint training; adapting vision only means an adapted image
embedding is **not comparable** with a frozen text embedding — the dot product
still returns a number, and that number means nothing.

> **transformers v5 note.** v4 returned a plain tensor from
> `get_*_features()`; v5 returns `BaseModelOutputWithPooling` whose
> `pooler_output` is *already* projected. Projecting again fails loudly for
> vision but **succeeds silently** for text on ViT-B/32 (512→512).
> `_as_embedding()` compares against `config.projection_dim` to catch it.

### [3] Linear probe — ~1,536 numbers

Logistic regression over the embeddings, in `backend/app/probe.npz`. The
**label** comes from `argmax`; the **continuous wetness** is still needed
because the temporal layer works on slope and a categorical label has no slope.

### [4] Temporal layer — where "drying" comes from

`backend/app/engine/temporal.py`, per session:

1. **EMA smoothing** (α = 0.45) — raw scores are too noisy for a usable slope
2. **Least-squares slope** over the last 5 frames, fitted rather than
   `last − first`. The **full** window is required — a partial fit measures the
   EMA settling, which once produced a phantom DRYING on twelve soaking frames
3. **Trend** — `DRYING` if slope ≤ −2.5 **and** wetness is above 45
4. **The four labels** — state gives three, direction gives the fourth
5. **Asymmetric hysteresis** — improving needs **3** consecutive frames,
   worsening only **2**. Slicks too early is a crash; slicks too late costs
   seconds. Direction is judged by `LABEL_SEVERITY`, not string equality
6. **Confidence gating** — below `CONFIDENCE_MIN = 0.50` a frame still nudges
   the score at half weight but cannot move the label

### [5] Suggestion — deterministic, deliberately small

12 headlines keyed by `(label, trend)`, plus a safety check against
`TIRE_BAND`. Safety is never traded against lap time: an earlier version had no
branch for "slicks on a wet track" and advised staying out to save a pit stop —
on a car with no tread in standing water.

**Live weather** comes from Open-Meteo (free, no key) using each circuit's
coordinates, labelled `live` / `operator` / `typical` on screen. If the feed is
unreachable the UI says so rather than showing a guess as a measurement.

---

## Results

All numbers are **leave-one-race-out**: every prediction comes from a model
that never saw a single frame of that race.

| | 96 frames, 6 races | 199 frames, 8 races | + LoRA |
|---|---|---|---|
| **3-class** | 0.365 | 0.487 | **0.573** |
| **dry frames kept dry** | **3.2%** | 32.9% | **60.0%** |
| wet vs not-wet | 0.854 | 0.799 | 0.819 |
| damp recall | — | 36% | 30% |

Per frame on a laptop CPU: ~110 ms frozen, ~156 ms with the adapter.

Two things drove the improvement, in this order:

1. **Fixing the dataset.** Dry-frames-kept-dry went 3.2% → 32.9% with *no
   model change at all*.
2. **Then** fine-tuning, which was worth +0.111 3-class — the first change in
   the project to beat the frozen baseline, and only possible once the data
   could support it.

> **On the 83.3% in earlier notes:** that was *random* 5-fold cross-validation.
> Frames from one race are near-duplicates, so a random split trains on
> siblings of its own test data. Holding out entire races dropped it to 36.5%.
> Every number above is venue-held-out.

---

## The confound, and how it was found

The system called dark-but-dry track damp: rubbered-in racing lines, overcast
days, floodlit night races. The cause turned out to be in the labels, and
finding that took ruling out everything else.

**The measurement that started it.** Under leave-one-race-out, **29 of 31 dry
frames were called damp — identically by a linear head, a 64-unit MLP, a
256-unit MLP and an RBF kernel.** Four capacities, the same 29 mistakes. That
is not a model that lacks power.

| hypothesis | test | result |
|---|---|---|
| the head is too weak | 4 classifier capacities | **No** — all made identical errors |
| CLIP confuses dark with wet | CLAHE lightness equalisation | **No** — dry accuracy moved 0.0 points, and wet-vs-not-wet *lost* 7 |
| the venue offset buries it | per-venue mean centring | **Untestable** — with single-class venues the venue mean *is* the class mean |
| the labels confound it | rebuild the dataset | **Yes** |

The original 96 frames came from six races, and **five were single-class**. Dry
came from Singapore, Austin and Vegas; damp from Monaco and São Paulo. So *"is
this damp"* and *"is this Monaco"* were the same question, and the model learned
venue identity. No architecture can undo that.

The fix was three race videos processed with `tools/extract_frames.py`, giving
199 frames across 8 races with **three venues holding several conditions each**
— the first time the dry/damp boundary was measurable at all.

**Five shortcut-confounds were caught this way**, each of which produced a
metric that looked like a triumph while the model was broken:

| # | shortcut | how it showed up |
|---|---|---|
| 1 | venue identity | 29/31 dry frames → damp, at every capacity |
| 2 | venue centring degeneracy | dry-kept-dry "improved" to 35% — which is chance |
| 3 | dataset source | BDD frames in one class only → 100% of frames predicted that class |
| 4 | off-by-one class mapping | rainy road labelled damp → the model never emitted wet |
| 5 | encoder leakage | an adapter trained on all frames scored 0.940 "held-out" |

`tools/_leakcheck.py` now guards every leave-one-race-out tool against #5.

---

## What was tried and removed

Each of these was physically plausible. Each was removed by measurement.

| Feature | Why it went |
|---|---|
| **SegFormer segmentation** | 290 ms/frame — 73% of the pipeline — and **0% road found** on real broadcast frames |
| **Darkness** | Scored a bone-dry photo of fresh dark tarmac at **84.5/100** |
| **Texture (Laplacian)** | Two equally wet frames: **25.9 and 487.1**. It measured motion blur |
| **Specular reflection** | **17.5 / 1.0 / 23.3** across three equally wet frames — broadcast auto-exposure |
| **Spatial sub-bands** | Scored *worse* than the full crop; monotonicity at chance |
| **Auto-ROI detection** | Boxed a car's engine cover and read **WET 75**. Kept as an experimental toggle for fixed cameras |
| **CLAHE** | Dry accuracy unchanged, wet-vs-not-wet **−7 points** |
| **CV feature fusion** | Re-tested on 199 frames: **0.447 vs 0.462 baseline** |
| **DINOv2 backbone** | Re-tested on 199 frames: **0.467 vs 0.462**. Tied |
| **flan-t5 phrasing** | Wrote *"The weather is a bit dry and wet"* three times in one message |

The classical CV features still compute and are reported as diagnostics. They
do not vote.

---

## Repository layout

```
backend/app/
  config.py              every tunable, with the measurement behind it
  main.py                FastAPI app, routes, model lifespan
  probe.npz              the trained classifier
  lora_adapter/          optional LoRA weights (~2.4 MB)
  models/
    clip_scorer.py       embeddings, probe, LoRA loading, crop verifier
    cv_features.py       classical CV signals (reported, not scored)
    phrasing.py          flan-t5 message writer with guards (off by default)
    segmentation.py      SegFormer wrapper — disabled, kept as the record
  engine/
    temporal.py          EMA, slope, hysteresis, confidence gating, evidence
    suggestion.py        tyre safety bands, urgency, headlines
    weather.py           Open-Meteo live conditions, honest fallback

frontend/src/            React dashboard, F1 broadcast styling

tools/
  extract_frames.py      turn race video into labelled frames
  audit_dataset.py       is the dataset able to answer the question?
  train_probe.py         fit the linear probe
  validate_loro.py       leave-one-race-out validation
  test_capacity.py       is the encoder the bottleneck, or the head?
  test_upgrades.py       re-test every proposed upgrade on current data
  test_roi.py            which ROI preset actually scores best
  verify_lora.py         is the adapter live in the real app path?
  _leakcheck.py          refuses to report contaminated numbers
  finetune_f1_lora.ipynb LoRA fine-tune, leave-one-race-out (Colab)
  finetune_lora.ipynb    LoRA on public road datasets (Colab)
  train_probe_v2.ipynb   retrain on RSCD/RoadSaW + acceptance gates (Colab)

circuits.json            24 circuits, every field marked with its provenance
PRESENTATION.md          demo run-sheet
```

---

## Building a dataset

The single most valuable property is **one venue containing several
conditions** — that is what makes dry-vs-damp measurable.

```bash
# 1. see where the race changes condition
python tools/extract_frames.py monaco2023.mp4 --venue monaco2023 --survey

# 2. extract each phase separately
python tools/extract_frames.py monaco2023.mp4 --venue monaco2023 \
    --label dry --start 0 --end 280 --every 4 --max 30 --verify
python tools/extract_frames.py monaco2023.mp4 --venue monaco2023 \
    --label wet --start 310 --end 480 --every 4 --max 30 --verify

# 3. check the dataset can answer the question BEFORE training
python tools/audit_dataset.py

# 4. train and measure
python tools/train_probe.py
python tools/validate_loro.py
```

`extract_frames.py` does four things that hand-screenshotting cannot: it names
files by venue so leave-one-race-out works, rejects near-duplicate frames,
discards non-track shots with CLIP, and detects camera type so onboard and
trackside stay balanced across classes.

`audit_dataset.py` prints a venue × class matrix and refuses to bless a set
where no venue holds more than one class.

---

## API reference

Base URL `http://127.0.0.1:8000`. Interactive docs at `/docs`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | model status, scoring method |
| `POST` | `/api/analyze/image` | **the main endpoint** — frame in, label + trend + suggestion out |
| `POST` | `/api/strategy` | recompute the tyre call without re-uploading |
| `POST` | `/api/reference` | mark the last frame as known-dry |
| `GET` | `/api/weather/{circuit}` | live conditions at the circuit |
| `GET` | `/api/circuits` | circuit list |
| `POST` | `/api/session/reset` | clear a session |
| `GET` | `/api/session/{id}` | full history and chart data |
| `POST` | `/api/roi/suggest` | experimental track detection |
| `POST` | `/api/debug/mask` | the analysed region as a PNG — **look here first when a score seems wrong** |

---

## Known limitations

**Damp is the ceiling.** 50 frames, and recall sits at 30–36% however it is
measured. It is the middle of a continuum labelled by timestamp, and the
boundary is genuinely fuzzy in footage. More damp frames would move this where
no architecture has.

**Night races remain the weak venue.** Floodlights create real specular
reflections on *dry* asphalt. The system degrades honestly: confidence drops
and the label is held rather than guessed.

**Four venues are still single-class** (`saopaulo2024`, `singapore2017`,
`singapore2025`, `us2025`, `vegas2025`). They drag the pooled number down
without being informative; one more condition at any of them is worth more
than a new venue.

**Absolute scores do not transfer between circuits.** The deployment answer is
the dry-reference button, not a global threshold.

**The wetness scale is not calibrated to a physical quantity.** It is 0–100,
not millimetres of water film.

**No dry-line detection.** The racing line dries first and that *is* the real
F1 decision — but the spatial-band experiment showed no usable signal, so the
feature was not built.

---

## Troubleshooting

**`LoRA adapter failed to load`** — the message names the reason. Missing
`peft`, missing `backend/app/lora_adapter/`, or a version mismatch. It always
falls back to frozen CLIP, so the app keeps working; `USE_LORA = False` silences
it.

**Everything reads WET, or everything reads DRY** — check the ROI first. Open
`/api/debug/mask`, or just look at the white box on screen.

**Frontend loads but shows "backend offline"** — the backend is still loading
models (first run downloads ~600 MB).

**A tool prints `USE_LORA=True - CONTAMINATED`** — the shipped adapter trained
on every frame, so leave-one-race-out through it is meaningless. Set
`USE_LORA = False` and re-run.

**Video mode samples nothing** — sampling is skipped while paused, by design.

---

## Licence

Source code is **[Apache 2.0](LICENSE)**. Third-party components keep their own
terms, listed in [NOTICE](NOTICE) — CLIP is MIT, flan-t5 Apache-2.0, Open-Meteo
CC BY 4.0.

**RSCD and RoadSaW are non-commercial licences.** They are used only by the
Colab notebooks for training experiments; no image or weight derived from
either is committed here.

---

## Against the brief

- **Frontend and backend** — React dashboard and FastAPI service over HTTP, not a notebook
- **Uses the Hugging Face Hub** — CLIP for vision, flan-t5-small available for language, both local
- **All four labels** — Dry, Damp and Wet from the classifier, Drying from the temporal layer
- **Trend graph** — raw and smoothed series with threshold bands and label-change markers
- **Suggestion message** — deterministic, safety-first, with the reasoning shown
- **Balanced difficulty** — a foundation model adapted on our own labelled data, a validated temporal layer, and a deterministic decision layer. Not one ready-made call; nothing from scratch.
