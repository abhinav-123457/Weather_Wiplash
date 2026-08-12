"""
config.py - every tunable number in one place.

Nothing here is sacred. These are starting points chosen so the system behaves
sensibly out of the box; RoadSaW calibration will replace the threshold and
temperature values with measured ones.
"""

# --------------------------------------------------------------------------
# Models (both pulled from the Hugging Face Hub)
# --------------------------------------------------------------------------
CLIP_MODEL = "openai/clip-vit-base-patch32"

# --------------------------------------------------------------------------
# Segmentation: DISABLED, on evidence.
#
# Tested against four real F1 broadcast frames (Monaco onboard, Sainz onboard,
# Monaco trackside). SegFormer/ADE20k found 0% road in every single one,
# returning instead:
#
#     wall 92.5%, signboard 6.5%
#     building 65.2%, wall 13.2%, sky 5.3%
#     building 57.9%, trade name 9.6%, bus 9.3%
#
# It reads circuit barriers and Monaco architecture as buildings and never
# recognises the track surface. Cost: 290ms/frame - 73% of the pipeline - for
# a 0% hit rate. Every frame fell back to the ROI crop anyway.
#
# Disabling it takes a frame from ~0.40s to ~0.11s, roughly 3.6x faster, and
# removes a silent failure mode. The operator defines the track region via
# ROI instead, which is what was actually happening regardless.
#
# The Hugging Face Hub requirement is unaffected: CLIP is a Hub model, and
# the published dataset satisfies it a second time.
#
# If you want the segmentation overlay back, try the CITYSCAPES checkpoint
# rather than ADE20k - Cityscapes is trained on road-facing dashcam footage,
# which is far closer to onboard F1 than ADE20k's general scene mix.
# --------------------------------------------------------------------------
USE_SEGMENTATION = False
SEG_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
SEG_MODEL_ALT = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

# Verified on the demo machine: ADE20k class 6 is "road". Still looked up at
# startup rather than hardcoded - label order is a property of the checkpoint.
#
# Deliberately permissive. ADE20k is trained on street scenes, where a dark
# mirror-like surface genuinely IS water - so it labels wet asphalt "water"
# and returns 0% road on exactly the frames this project exists to analyse.
# Accepting the ground-like classes costs little once an ROI has already
# restricted us to the track, and it stops the wettest frames scoring zero.
ROAD_LABEL_KEYWORDS = (
    "road", "path", "water", "earth", "ground",
    "floor", "runway", "sidewalk", "pavement", "field",
)

# Below this fraction of road pixels there is not enough surface to judge,
# so the API abstains rather than returning a confident number computed from
# almost nothing. Lowered from 0.05 - real broadcast frames legitimately show
# only a slice of track, and abstaining on those helps no one.
MIN_ROAD_COVERAGE = 0.02

# Default crop per camera type, as (x0, y0, x1, y1) fractions.
#
# An onboard frame is mostly car - halo, mirrors, bodywork, front wheels.
# ADE20k has no class for any of that, so SegFormer scatters those pixels
# across whatever looks nearest and road coverage collapses below threshold.
# Cropping to the upper band before segmenting fixes it at source.
# With segmentation disabled, THE ROI IS THE MASK. Everything inside it is
# scored as track surface, so a loose crop that includes barriers, crowd or
# sky will corrupt the reading. This is now the single most important control
# in the system and belongs in the UI as a draggable box.
#
# These presets are only sensible starting points - the track sits in a
# different part of the frame in almost every shot.
ROI_PRESETS = {
    # Avoids the car body along the bottom and the skyline along the top.
    "onboard":   [0.08, 0.14, 0.92, 0.55],
    # Broadcast trackside shots usually put the surface across the middle.
    "trackside": [0.03, 0.20, 0.97, 0.90],
}

# --------------------------------------------------------------------------
# THE DRAWN BOX IS NOT WHAT CLIP SAW - until this flag was added.
#
# CLIPProcessor resizes the SHORTEST side to 224 and then centre-crops to
# 224x224. The ROI presets are wide, flat bands, so most of the drawn width
# was discarded before the model ever looked at it:
#
#     preset      drawn (1920x1080)   aspect   width CLIP kept
#     onboard     1613 x 443          3.6:1    ~25%
#     trackside   1805 x 756          2.4:1    ~40%
#
# Three-quarters of an onboard selection was being thrown away, which is
# also why the score was so sensitive to exactly where the box sat: moving
# it shifted which quarter survived.
#
# Resizing the crop to a square first means the WHOLE selection reaches
# CLIP. It distorts the aspect ratio, but this is a material judgement -
# is this asphalt wet - and material texture survives stretching far better
# than it survives being deleted.
#
# Precedent: tools/inspect_bands.py already did exactly this for the band
# experiment, for exactly this reason.
#
# IF SCORES LOOK WRONG AFTER THIS CHANGE, set it to False - that restores
# the previous behaviour exactly. Verify with the Test ROI button on one
# known-dry and one known-wet frame before trusting a session.
# --------------------------------------------------------------------------
ROI_TO_SQUARE = True
ROI_SQUARE_SIZE = 224          # CLIP ViT-B/32's native input resolution

# SegFormer is 73% of per-frame cost (290ms of 397ms measured). If you need
# speed, shrink this - do not optimise CLIP, it is only 103ms.
SEG_INPUT_SIZE = 512


# --------------------------------------------------------------------------
# CLIP scoring
# --------------------------------------------------------------------------
# State prompts anchored to wetness values. There is deliberately NO "drying"
# prompt: drying is a trend, computed from slope across frames. Asking a
# single image whether the track is drying reintroduces the exact flaw this
# architecture exists to avoid.
# PROMPT ENSEMBLING - several prompts per class, embeddings averaged.
#
# A single prompt per class scored 61% on 96 labelled frames, and 16 of 31
# dry frames were called damp. Eleven of those were SINGAPORE - a night race.
# The old dry prompt said "no reflections", but a dry track under floodlights
# is full of reflections, so CLIP had no "dry at night" reference to reach
# for and picked damp instead.
#
# Each group therefore covers daylight AND artificial light, near AND far.
# Embeddings are L2-normalised, averaged within the group, then renormalised,
# giving one robust direction per class instead of one brittle sentence.
#
# Still no "drying" group. Drying is a trend across frames and cannot be
# described in a single image - that is the premise of the whole system.
PROMPT_GROUPS = {
    "dry": [
        "a dry asphalt racetrack in daylight, matte grey surface",
        "a dry asphalt racetrack at night under floodlights, no water",
        "dry racing asphalt, rough dry texture, no water anywhere",
        "a dry racetrack with a clear racing line and dusty kerbs",
        "dry dark asphalt, no puddles, no water film",
    ],
    "damp": [
        "a damp asphalt racetrack, darkened by moisture, no standing water",
        "a racetrack with a faint damp sheen but no puddles",
        "a racetrack surface drying after rain, patchy moisture",
        "slightly wet asphalt at night, damp but no standing water",
    ],
    "wet": [
        "a wet asphalt racetrack with standing water and mirror reflections",
        "a racetrack in heavy rain with puddles and spray from cars",
        "a soaked racetrack at night, water reflecting the lights",
        "a flooded racetrack with deep standing water",
    ],
}

# Wetness value anchoring each class. The reported score is the
# probability-weighted average of these.
PROMPT_VALUES = {"dry": 0.0, "damp": 50.0, "wet": 100.0}

# --------------------------------------------------------------------------
# SCENE CONTEXT - scored on the FULL frame, not the ROI crop.
#
# The surface classifier sees only asphalt. This sees sky, shadows, crowd
# and spray - genuinely independent evidence, which is exactly what the
# three failed CV features lacked (all three re-read the same pixels).
#
# The physical argument: hard-edged shadows mean direct sun, and sun does
# not coexist with rain. Umbrellas and ponchos in a crowd mean rain. Neither
# is visible inside a crop of asphalt.
#
# REPORTED ONLY, NOT FUSED. Three previous signals looked physically sound
# and turned out to measure camera artefacts instead. This one gets measured
# against labelled data before it is allowed to move a label.
# --------------------------------------------------------------------------
USE_CONTEXT = True

CONTEXT_PROMPT_GROUPS = {
    "sunny": [
        "a bright sunny day at a racetrack, hard shadows on the ground",
        "clear blue sky over a motor racing circuit",
        "strong sunlight, sharp shadows, dry weather",
    ],
    "overcast": [
        "an overcast grey day at a racetrack, flat light, no shadows",
        "cloudy sky over a racing circuit, dull light",
        "grey clouds, no rain falling, no shadows",
    ],
    # Night needed its own group. Without it a floodlit race matched
    # 'overcast' - no sun, no blue sky - and inherited a 45/100 rain
    # likelihood, so the context signal actively pushed dry night races
    # toward damp. That made the system's worst case worse.
    "night_dry": [
        "a night race under floodlights, dry track surface",
        "a floodlit motor racing circuit at night, no rain",
        "artificial stadium lighting at night, dry asphalt",
        "night race, dry track, light reflecting off dry tarmac",
    ],
    "raining": [
        "heavy rain falling at a motor racing circuit",
        "rain and spray in the air, cars throwing water",
        "wet weather race, spectators under umbrellas",
        "a night race in the rain, wet reflections and spray",
    ],
}

# Rough likelihood-of-water each context implies, 0-100. Used only to turn
# the context distribution into one comparable number.
CONTEXT_VALUES = {
    "sunny": 0.0,
    "night_dry": 15.0,
    "overcast": 45.0,
    "raining": 100.0,
}

# Surface and context disagreeing by more than this is worth surfacing: a
# frame reading WET under hard sunlight is one of them being wrong, and a
# system that says so is more trustworthy than one that quietly picks.
CONTEXT_DISAGREEMENT = 45.0

# --------------------------------------------------------------------------
# Linear probe - a logistic regression over frozen CLIP embeddings.
#
# Measured on 96 labelled frames, 5-fold cross-validated:
#
#     zero-shot prompts   54%
#     linear probe        83%      dry 94% | damp 61% | wet 94%
#                                  wet vs not-wet: 90%
#
# CLIP is untouched - the probe is 1,536 numbers in probe.npz. Prompts stay
# in the file as the fallback when no probe is present, and as the thing the
# probe is measured against.
#
# Damp sits at 61% and errs in BOTH directions (4 called dry, 8 called wet).
# That is not a bug to fix: damp genuinely lies between two states and often
# is not distinguishable in a single frame. The temporal layer resolves it -
# a damp reading falling from wet is drying, one rising from dry is wetting.
# --------------------------------------------------------------------------
USE_PROBE = True
PROBE_PATH = "probe.npz"

# Optional SECOND-OPINION probe - never votes.
#
# probe_road.npz is trained on ~1M public road images (RSCD + RoadSaW) and
# ZERO F1 frames. Measured venue-held-out against our 96 F1 frames: ~49%
# 3-class - above the F1-only probe's 36.5% - but with a hard blind spot
# measured in the same run: floodlit night tracks read as standing water
# (dry Singapore frames scored 94-100). Both datasets are daytime; the
# model has never seen floodlights.
#
# Mixing the domains into ONE probe was tried and made both worse: every
# mixture (10/30/50% F1 weight) underperformed BOTH endpoints, across two
# independent runs. A single 1,536-parameter hyperplane cannot hold "glare
# means wet" for daytime roads and "glare means dry floodlights" for night
# races at the same time.
#
# So the road model is kept as an independent second opinion: displayed
# beside the decision, never fused into it - the same rule scene context
# follows. Absent file = feature off.
ROAD_PROBE_PATH = "probe_clip_both.npz"

# PRIMARY TUNING KNOB - not a detail.
# Measured CLIP similarities span only ~0.04 between prompts, so this value
# decides how much of that narrow gap becomes confident probability.
# Too low  -> noise becomes certainty.
# Too high -> every frame collapses to the mean.
CLIP_TEMPERATURE = 0.01


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
# CLIP ONLY - decided by measurement, not preference.
#
# Three classical CV signals were built and each tested against real F1
# broadcast frames. All three tracked camera behaviour rather than water:
#
#   darkness  swamped by asphalt colour. A dry dark-tarmac photo scored 84.5.
#   texture   swamped by motion blur. Two equally wet Monaco frames gave
#             Laplacian variances of 25.9 and 487.1 - a 19x swing.
#   specular  swamped by auto-exposure. Wet road reflects diffuse overcast
#             sky at ~140 lightness and never crosses the >200 threshold.
#             Scored 17.5 / 1.0 / 23.3 across three equally wet frames.
#
# Net effect at CV_WEIGHT 0.15: every wet frame was pulled DOWN 5-12 points,
# and at 0.40 it flipped Monaco from WET to DAMP. CV did not once improve a
# reading.
#
# The features still compute and are reported under signals.cv as
# diagnostics, so RoadSaW calibration can revisit them with ground truth.
# They simply do not vote until they earn it.
CLIP_WEIGHT = 1.00
CV_WEIGHT = 0.00

# Weights within the classical-CV signal itself.
#
# Absolute darkness is NOT here, and that is deliberate. It originally held
# 0.50 and scored a dry dark-tarmac photo at 84.5/100 wetness. Dark asphalt
# and wet asphalt look alike in brightness alone; the discriminators are
# reflectivity and surface texture, both of which survive a change of
# asphalt colour.
CV_SPECULAR_WEIGHT = 0.60     # water is mirror-like
CV_SPRAY_WEIGHT = 0.40        # trackside only, renormalised when absent

# TEXTURE IS DISABLED - it measured camera sharpness, not wetness.
#
# Two equally wet Monaco frames returned Laplacian variances of 25.9 and
# 487.1, a 19x swing, because one was motion-blurred and the other sharp.
# Scored as wetness that became 100 and 0. Motion blur, focus and codec
# artefacts dominate high-frequency detail far more than a water film does,
# and broadcast footage varies wildly in all three between frames.
#
# Kept in the response as a raw diagnostic (laplacian_var) so calibration
# can revisit it, but it no longer votes. Salvaging it would mean
# normalising road texture against off-road texture in the same frame to
# cancel out global blur - worth trying only after everything else works.
CV_TEXTURE_WEIGHT = 0.0
TEXTURE_VAR_WET = 40.0
TEXTURE_VAR_DRY = 400.0


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
# Fitted to six measured broadcast frames (CLIP-only scores):
#
#   DRY   Suzuka start      16.9      WET   Zandvoort   48.7  (draining)
#         Silverstone       25.9            Sainz       77.5
#         Suzuka trackside  41.5            Monaco      80.7  (downpour)
#
# The clusters do not overlap, but the margin is only 7.2 points
# (41.5 -> 48.7), so these cuts sit mid-gap rather than anywhere generous.
#
# The old 25/55 sliced through the middle of the dry cluster, which is why
# genuinely dry tracks kept coming back DAMP. CLIP was right; the thresholds
# were wrong.
#
# SIX FRAMES IS THIN. These hold for everything measured so far, but RoadSaW
# calibration with sensor ground truth should replace them - that is the
# single highest-value thing left to do to this file.
DRY_THRESHOLD = 45.0
DAMP_THRESHOLD = 65.0
WET_THRESHOLD = 85.0      # reserved for a future "very wet" split

# Never emit DRYING on an already-dry track: a small negative slope at
# wetness 12 is noise, not a trend. Tracks DRY_THRESHOLD.
DRYING_MIN_WETNESS = 45.0


# --------------------------------------------------------------------------
# Dry-reference calibration (per camera, per session)
#
# LORO score inspection measured a per-venue offset of 16-28 wetness points -
# the same size as the between-class gap - so no global threshold survives a
# change of venue or camera. The deployment answer is a one-time reference:
# the operator analyses a frame they KNOW shows dry track and marks it. The
# gap between that reading and what a well-calibrated dry track reads becomes
# a constant subtracted from every later frame in the session.
#
# It is a SHIFT, not a rescale: one reference point determines one constant.
# While a reference is active the state comes from thresholds on the adjusted
# score instead of the probe's argmax - the probe's absolute calibration is
# exactly the thing the operator just corrected.
# --------------------------------------------------------------------------
# What a dry track reads when the classifier is in-domain. Probe-scored dry
# frames cluster low but not at zero (the six-frame table above: 16.9-41.5
# CLIP-only; probe outputs sit lower). Mid-teens is the working figure.
DRY_REFERENCE_ANCHOR = 12.0

# |offset| beyond this means the "known dry" frame probably was not dry -
# refuse it rather than silently mis-calibrating the whole session.
REFERENCE_OFFSET_LIMIT = 45.0


# --------------------------------------------------------------------------
# AUTO-ROI: EXPERIMENTAL, OFF BY DEFAULT - measured worse than the presets.
#
# Three attempts, three failures on real Monaco/Singapore onboard footage:
#
#   1 bounding box of all road pixels   spanned the car between two road
#                                       regions; scored an Aston Martin
#                                       engine cover as WET 75
#   2 density-trimmed single box        Cityscapes' bottom-of-frame prior
#                                       labels dark bodywork "road", so the
#                                       biggest region WAS the car
#   3 candidates + CLIP verification    better, but still boxed a helmet and
#                                       held a stale car box across a cut
#
# The honest verdict: for CUT BROADCAST footage, the camera preset plus one
# drag beats automatic detection. The code stays because the idea is sound
# for the actual deployment case - a FIXED trackside camera, where the view
# never cuts and detection runs once at install - but it must not be the
# default path, and the UI labels it experimental.
#
# This is the sixth feature measured and demoted rather than argued for:
# segmentation, darkness, texture, specular, sub-bands, auto-ROI.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Auto-ROI crop verification
#
# The Cityscapes segmenter carries a bottom-of-frame prior and labelled an
# onboard car's dark bodywork "road" (measured live: the box sat on a
# Mercedes cockpit reading WET at conf 0.72). So no detector box is trusted
# until CLIP agrees the crop LOOKS like track: a coarse, contrastive
# choice - asphalt vs cockpit vs graphics vs surroundings - which is
# exactly the kind of judgement CLIP is reliable at. Candidates that lose
# are rejected and the next one is tried.
# --------------------------------------------------------------------------
TRACK_VERIFY_PROMPT_GROUPS = {
    "track": [
        "a patch of asphalt road surface",
        "grey tarmac racetrack surface with painted lines",
        "wet asphalt surface of a road",
    ],
    "car": [
        "the cockpit of a formula 1 car with halo and steering wheel",
        "glossy bodywork of a racing car, close up",
        "the rear wing and tyres of a racing car",
    ],
    "graphics": [
        "television broadcast graphics and text overlay",
        "a plain black bar with no content",
    ],
    "surroundings": [
        "crowd in a grandstand at a race",
        "barriers, fences and buildings beside a racetrack",
        "sky and trees",
    ],
}


# --------------------------------------------------------------------------
# Auto-ROI follow (live video)
#
# Broadcast footage cuts between shots, and a box placed for one shot is
# wrong for the next. Every frame already gets a full-scene CLIP embedding
# (scene context uses it), so a cut is detectable for free: cosine
# similarity between consecutive frames' scene embeddings below this value
# means the shot changed, and only then does the track detector re-place
# the box. Consecutive frames from one camera sit ~0.9+; hard cuts fall
# well under 0.8.
# --------------------------------------------------------------------------
SHOT_CHANGE_SIM = 0.80


# --------------------------------------------------------------------------
# Temporal
# --------------------------------------------------------------------------
# Higher = reacts faster, lower = smoother.
#
# 0.30 was measurably too slow for short sequences. On a 15-frame run, a
# step from 84 to 17 left the smoothed value at 60 three frames later - a
# third of the sequence spent catching up, and the label never reached DRY
# before the frames ran out.
#
# 0.45 converges in roughly half the frames while still absorbing the
# single-frame spikes that hysteresis is there to survive. If you ever feed
# this a long continuous stream (hundreds of frames), drop it back toward
# 0.30 for better noise rejection.
EMA_ALPHA = 0.45
SLOPE_WINDOW = 5          # frames in the least-squares fit

# Wetness points per FRAME, so these depend on how far apart your frames are.
# Frames a few minutes apart across a drying race produce large per-frame
# changes; frames seconds apart produce small ones. Retune if you change the
# sampling interval - this is the knob most likely to need it.
# Raised from +/-1.5 after a FALSE DRYING on real footage: twelve Monaco
# onboard frames, all soaking (wetness 83-90, no drying at all), fitted a
# -2.98 slope and committed DRYING for three frames before reverting.
#
# ATTRIBUTION, since two fixes went in together and only one was needed:
#
#     window   threshold   stable-wet        drying
#     partial   -1.5       WET->DRYING->WET  correct   <- the bug
#     partial   -2.5       WET               correct
#     full      -1.5       WET               correct
#     full      -2.5       WET               correct
#
# The ROOT CAUSE was the partial slope window (see _slope in temporal.py),
# which fitted the EMA settling rather than the track. That fix alone is
# sufficient.
#
# This threshold is kept anyway as margin, at a known cost: drying slower
# than 2.5 wetness-points per frame is no longer flagged. That is an
# acceptable trade because slow drying is not an urgent tire call - the
# decision matters when conditions are moving fast.
#
# Both numbers are tuned to the noise of THIS classifier on broadcast
# frames. Better calibration data would justify lowering them again.
SLOPE_DRYING_THRESHOLD = -2.5
SLOPE_WETTING_THRESHOLD = 2.5

# ASYMMETRIC HYSTERESIS - the risk is not symmetric, so the delay must not be.
#
# Going to slicks too early on a damp track means a spin or a crawl:
# catastrophic. Going too late costs a couple of seconds a lap: recoverable.
# So improving conditions must prove themselves, while worsening conditions
# are acted on immediately.
#
# A symmetric rule also had a concrete bug. Requiring N consecutive IDENTICAL
# candidates meant a fast-worsening track (DRY -> DAMP -> WET) reset the
# counter at every step and never committed - the label sat on DRY through
# an entire simulated downpour.
# Below this the classifier is not discriminating and the frame must not be
# allowed to move the label. With three classes, pure chance is 0.33.
#
# Measured need: five Singapore night frames labelled dry scored 16.8, 23.0,
# 45.8, 49.9 and 62.5 with confidences of 0.74, 0.66, 0.37, 0.35 and 0.42.
# The three uncertain ones dragged a correctly-drying sequence back to DAMP
# and flipped the trend to WETTING. A system that says "I am not sure" is
# more useful, and more honest, than one that guesses with a straight face.
#
# Raised 0.45 -> 0.50 after a second live failure: six all-dry Singapore
# frames, where two consecutive glare misreads (raw 67.9 @ 0.55, then 68.1
# @ 0.48) hit the 2-frame worsening path and committed a WET banner that
# three correct dry frames then couldn't undo fast enough. At 0.50 the
# 0.48 frame is held, the pending count never reaches 2, and the banner
# stays DRY through the whole sequence. Cost: frames at 0.45-0.50 now hold
# the label for one extra frame - acceptable, since a commit that weak was
# exactly what produced the false alarm.
CONFIDENCE_MIN = 0.50

# Low-confidence frames still nudge the score, at reduced weight - they are
# weak evidence, not no evidence.
LOW_CONFIDENCE_ALPHA_SCALE = 0.5

HYSTERESIS_FRAMES = 3           # improving: hold until sustained
HYSTERESIS_FRAMES_WORSENING = 2 # worsening: act fast, but not on one frame
#
# Why 2 and not 1: at 1, a single noisy WET frame in a damp sequence flipped
# the banner to WET and held it for five frames - noise turned into a false
# alarm. Two frames filters single-frame spikes while still reacting to
# genuine rain a full frame earlier than the improving path.

# Ordering by how much water is present. Used only to decide which of the
# two delays above applies. DRYING sits between DRY and DAMP: water is still
# there, but leaving.
LABEL_SEVERITY = {"DRY": 0, "DRYING": 1, "DAMP": 2, "WET": 3}


# --------------------------------------------------------------------------
# Bands / dry-line detection
# --------------------------------------------------------------------------
BAND_COUNT = 6
BIMODALITY_MIN = 0.25


# --------------------------------------------------------------------------
# Video
# --------------------------------------------------------------------------
# Measured 0.40 s/frame -> 40 frames is ~16s of processing. Long enough for
# the chart to visibly build, short enough to hold a room's attention.
VIDEO_FRAME_BUDGET = 40


# --------------------------------------------------------------------------
# Live weather (Open-Meteo, free, no API key)
#
# Replaces the "typical race-day conditions" fallback as the default source:
# a guessed 32C/75% on screen was indefensible next to real measurements.
# Precedence per field: operator-supplied > live > typical - and the
# response always says which one was used.
# --------------------------------------------------------------------------
USE_LIVE_WEATHER = True
WEATHER_TTL_S = 600      # one fetch per circuit per 10 min; failures retry in 60s


# --------------------------------------------------------------------------
# Strategy engine
# --------------------------------------------------------------------------
# Wetness points removed per minute at REFERENCE_VPD, no wind, on a permanent
# circuit in full sun. A tuning constant, not a measured physical quantity -
# say so if asked.
BASE_DRYING_RATE = 2.5
REFERENCE_VPD = 12.0          # hPa, moderate conditions - the 1.0x point
WIND_FACTOR_PER_MS = 0.04     # wind carries saturated air off the surface

# Circuit contribution to drying.
#
# CLIMATE IS DELIBERATELY ABSENT even though circuits.json carries it: the
# caller supplies real track temp, air temp and humidity, so a climate factor
# here would count the same effect twice.
SURFACE_DRAINAGE = {"street": 0.60, "semi_permanent": 0.85, "permanent": 1.00}
SHADE_FACTOR = {"high": 0.75, "some": 0.90, "none": 1.00}

# Severity only nudges compound choice. It never touches drying rate or
# wetness detection - those are different physics.
SEVERITY_FACTOR = {"low": 0.70, "medium": 1.00, "high": 1.15, "very_high": 1.30}

# Tire-family boundaries on the wetness scale.
FULL_WET_THRESHOLD = 80.0     # standing water, aquaplaning risk
INTER_THRESHOLD = 45.0        # below this slicks are the faster tire

# The wetness band each tyre can actually be used in.
#
# This replaces a set of hand-written if/else branches that only covered the
# cases I happened to think of. It missed an obvious and dangerous one:
# slicks fitted on a wet track were treated as an ECONOMIC decision, so with
# few laps left the engine said "stay out - you would not recover the pit
# loss". Nobody stays on slicks in standing water to save twenty seconds.
#
# Stating the safe band for every tyre makes the gap impossible: any tyre
# outside its band is flagged, whether or not I anticipated that combination.
#
#   safe_max  above this the tyre cannot clear the water -> UNSAFE
#   safe_min  below this the tyre overheats with no water to cool it
# SLICK is the family key the UI sends; the individual compounds are kept so
# an API caller passing SOFT/MEDIUM/HARD is still covered. Omitting SLICK
# meant a family-level lookup found no band and silently assumed the tyre was
# fine - slicks at wetness 78 came back as merely "advisory".
TIRE_BAND = {
    "SLICK":    {"safe_min": 0.0,  "safe_max": 45.0},
    "SOFT":     {"safe_min": 0.0,  "safe_max": 45.0},
    "MEDIUM":   {"safe_min": 0.0,  "safe_max": 45.0},
    "HARD":     {"safe_min": 0.0,  "safe_max": 45.0},
    "INTER":    {"safe_min": 25.0, "safe_max": 80.0},
    "FULL_WET": {"safe_min": 55.0, "safe_max": 100.0},
}

# Decision precedence. Safety is never traded against lap time - the pit-loss
# economics only ever apply to an OPPORTUNITY, never to a hazard.
#   1 SAFETY       current tyre cannot handle the water
#   2 DEGRADATION  current tyre will destroy itself
#   3 OPPORTUNITY  a faster tyre exists; economics decide
#   4 STEADY       nothing to do

# Stint-length bands for choosing WHICH slick.
COMPOUND_LAPS_SOFT = 15
COMPOUND_LAPS_MEDIUM = 30

# Approximate lap-time advantage of slicks over inters on a dry line, used
# only in the endgame pit-loss calculation. An estimate - but the DECISION it
# drives is far less sensitive than the number looks, since it is compared
# against a ~20s pit loss and only flips near the very end of a race.
TIME_GAIN_SLICK_VS_INTER = 2.5   # seconds per lap


# --------------------------------------------------------------------------
# Suggestion phrasing - a Hugging Face text model writes the sentence.
#
# The DECISION stays in rules and stays deterministic: whether slicks in
# standing water are safe is not a matter of opinion, and generated text
# cannot be safety-checked before a user reads it.
#
# The PHRASING is generated, because turning structured facts into a natural
# radio message is what a small instruction-tuned model is genuinely good at,
# and nothing it writes can change what was decided.
#
# flan-t5-small: ~300MB, instruction-tuned, runs on CPU in well under a
# second. flan-t5-base reads better at ~1GB if the download is affordable.
# Any failure falls back to the deterministic template, so this can never
# take the system down.
# --------------------------------------------------------------------------
# OFF by default since 2026-08-13: observed live output was "The weather is
# a bit dry and wet" repeated three times - rejected by the guards now, but
# the deterministic templates are already good sentences and every word of
# them can be defended on stage. Set True to re-enable the model; the
# guards (repetition, prompt echo, tyre-mention) remain in place.
USE_PHRASING = False
PHRASING_MODEL = "google/flan-t5-small"
PHRASING_MIN_CHARS = 12
PHRASING_MAX_CHARS = 180