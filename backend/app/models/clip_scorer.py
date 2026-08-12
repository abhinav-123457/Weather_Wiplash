"""
clip_scorer.py - language-prompted wetness scoring.

CLIP is a vision-LANGUAGE model: we embed the image, embed natural-language
descriptions of track states, and compare them in a shared space. The prompts
are how the model is tuned - there is no training data and no fixed label set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from ..config import (CLIP_MODEL, CLIP_TEMPERATURE, CONTEXT_PROMPT_GROUPS,
                      CONTEXT_VALUES, PROBE_PATH, PROMPT_GROUPS,
                      PROMPT_VALUES, ROAD_PROBE_PATH,
                      TRACK_VERIFY_PROMPT_GROUPS, USE_CONTEXT, USE_PROBE)


def _as_embedding(out, model: CLIPModel, kind: str) -> torch.Tensor:
    """Normalise get_*_features output to a shared-space embedding.

    transformers v4 returned a plain tensor here; v5 returns a
    BaseModelOutputWithPooling. Verified on transformers 5.15.0: v5's
    pooler_output is ALREADY projected to config.projection_dim.

    Never project blindly. For vision, a second projection fails loudly
    (768x512 weight against a 512-d input). For text on ViT-B/32 the
    projection is 512->512, so double-projecting SUCCEEDS SILENTLY and
    produces wrong embeddings with a correct-looking shape. Only comparing
    against projection_dim catches that.
    """
    if torch.is_tensor(out):
        return out

    for attr in ("text_embeds", "image_embeds"):
        val = getattr(out, attr, None)
        if torch.is_tensor(val):
            return val

    pooled = getattr(out, "pooler_output", None)
    if torch.is_tensor(pooled):
        target = getattr(model.config, "projection_dim", None)
        if target is None or pooled.shape[-1] == target:
            return pooled
        proj = model.text_projection if kind == "text" else model.visual_projection
        return proj(pooled)

    raise TypeError(f"cannot extract embedding from {type(out).__name__}")


class ClipScorer:
    """Scores an image against wetness-anchored language prompts.

    The text tower runs ONCE, at construction. Prompts never change, so the
    63M-parameter text encoder has no business in the per-frame path.
    """

    def __init__(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL, use_safetensors=True)
        self.model.eval()

        self.class_names = list(PROMPT_GROUPS.keys())
        self.anchor_values = np.array(
            [PROMPT_VALUES[c] for c in self.class_names], dtype=np.float32)
        self.prompts = [p for c in self.class_names for p in PROMPT_GROUPS[c]]

        # Prompt ensembling: normalise each prompt embedding, average within
        # the class, renormalise. Averaging RAW embeddings would let longer
        # sentences dominate by magnitude; normalising first makes it a true
        # mean direction, so every phrasing contributes equally.
        group_embeddings = []
        with torch.no_grad():
            for cls in self.class_names:
                toks = self.processor(text=PROMPT_GROUPS[cls],
                                      return_tensors="pt", padding=True)
                emb = _as_embedding(self.model.get_text_features(**toks),
                                    self.model, "text")
                emb = emb / emb.norm(dim=-1, keepdim=True)
                mean = emb.mean(dim=0)
                group_embeddings.append(mean / mean.norm())

        self.text_embeddings = torch.stack(group_embeddings)

        # Crop verifier: is this box actually track surface, or the car's
        # bodywork / TV graphics the segmenter mistook for road? Same
        # ensembling recipe; built once at startup like everything else.
        self.verify_names = list(TRACK_VERIFY_PROMPT_GROUPS.keys())
        verify = []
        with torch.no_grad():
            for name in self.verify_names:
                toks = self.processor(text=TRACK_VERIFY_PROMPT_GROUPS[name],
                                      return_tensors="pt", padding=True)
                e = _as_embedding(self.model.get_text_features(**toks),
                                  self.model, "text")
                e = e / e.norm(dim=-1, keepdim=True)
                mean = e.mean(dim=0)
                verify.append(mean / mean.norm())
        self.verify_embeddings = torch.stack(verify).numpy()

        # Same ensembling recipe for the scene prompts.
        self.context_names = list(CONTEXT_PROMPT_GROUPS.keys())
        self.context_embeddings = None
        if USE_CONTEXT:
            ctx = []
            with torch.no_grad():
                for name in self.context_names:
                    toks = self.processor(text=CONTEXT_PROMPT_GROUPS[name],
                                          return_tensors="pt", padding=True)
                    e = _as_embedding(self.model.get_text_features(**toks),
                                      self.model, "text")
                    e = e / e.norm(dim=-1, keepdim=True)
                    mean = e.mean(dim=0)
                    ctx.append(mean / mean.norm())
            self.context_embeddings = torch.stack(ctx).numpy()

        # Optional linear probe. Absent or disabled -> prompts are used.
        self.probe = None
        if USE_PROBE:
            path = Path(__file__).resolve().parents[1] / PROBE_PATH
            if path.exists():
                d = np.load(path, allow_pickle=True)
                if not self._dim_ok(d, path.name):
                    pass                     # message printed by the check
                else:
                    self.probe = {
                        "coef": d["coef"], "intercept": d["intercept"],
                        "classes": [str(c) for c in d["classes"]],
                        "values": d["values"],
                        "cv_accuracy": float(d["cv_accuracy"]),
                    }
                    print(f"  probe loaded: {self.probe['cv_accuracy']:.1%} "
                          f"cross-validated")
            else:
                print(f"  USE_PROBE is on but {path.name} is missing - "
                      f"falling back to prompts")

        # Optional second-opinion probe, trained on public road datasets
        # with zero F1 frames. REPORTED, NEVER FUSED - it has a measured
        # night-race blind spot (see ROAD_PROBE_PATH in config), and mixing
        # the domains into one probe measurably hurt both. Same rule as
        # scene context: independent evidence earns a display row, not a
        # vote, until it wins on labelled data.
        self.road_probe = None
        road_path = Path(__file__).resolve().parents[1] / ROAD_PROBE_PATH
        if road_path.exists():
            d = np.load(road_path, allow_pickle=True)
            if self._dim_ok(d, road_path.name):
                self.road_probe = {
                    "coef": d["coef"], "intercept": d["intercept"],
                    "classes": [str(c) for c in d["classes"]],
                    "values": d["values"],
                }
                print("  road probe loaded (second opinion only - "
                      "never votes)")

    # ----------------------------------------------------------------------
    def _dim_ok(self, d, name: str) -> bool:
        """Refuse a probe whose coefficients do not match CLIP's embedding.

        The training notebook also produces probe_dinov2_*.npz files (384-d).
        Copied here by mistake, one would crash with a shape error on the
        FIRST FRAME - mid-demo - instead of at startup. Refusing loudly at
        load time turns that into a one-line fix before anyone is watching.
        """
        expect = int(getattr(self.model.config, "projection_dim", 512))
        got = int(d["coef"].shape[1])
        if got == expect:
            return True
        print(f"  ! {name} REFUSED: {got}-d coefficients vs CLIP's "
              f"{expect}-d embedding - this is a probe for a different "
              f"backbone (use a probe_clip_* file)")
        return False

    def _probe_scores(self, emb: np.ndarray,
                      probe: dict) -> tuple[float, np.ndarray, float, str]:
        """(wetness, probs, conf, label) for one probe on a ready embedding."""
        logits = emb @ probe["coef"].T + probe["intercept"]
        z = logits - logits.max()
        probs = np.exp(z)
        probs /= probs.sum()
        wetness = float(probs @ probe["values"])
        idx = int(probs.argmax())
        return wetness, probs, float(probs[idx]), probe["classes"][idx]

    def probe_from(self, emb: np.ndarray) -> tuple[float, np.ndarray, float, str]:
        """Main probe on an already-computed embedding.

        The LABEL comes from argmax, not from thresholding the score. The
        probe outputs class probabilities directly, so thresholds would only
        re-derive - less accurately - a decision it has already made.

        The continuous WETNESS is still needed: the temporal layer works on
        slope, and a categorical label has no slope.
        """
        return self._probe_scores(emb, self.probe)

    def verify_crop(self, image) -> dict:
        """Does this crop look like track surface? Returns the winner.

        A coarse contrastive choice - asphalt vs cockpit vs graphics vs
        surroundings - which is exactly what CLIP is dependable at, unlike
        the fine wet/damp distinction that needed a trained probe.

        Exists because the segmenter cannot audit its own mistake: the
        Cityscapes model's bottom-of-frame prior labelled an onboard car's
        dark bodywork "road", and the resulting box scored the cockpit.
        """
        emb = self.embedding(image)
        sims = emb @ self.verify_embeddings.T
        z = sims / CLIP_TEMPERATURE
        z = z - z.max()
        probs = np.exp(z)
        probs /= probs.sum()
        idx = int(probs.argmax())
        return {"kind": self.verify_names[idx],
                "confidence": round(float(probs[idx]), 3),
                "is_track": self.verify_names[idx] == "track"}

    def road_opinion_from(self, emb: np.ndarray) -> dict | None:
        """Second opinion on the SAME embedding - zero extra model cost."""
        if self.road_probe is None:
            return None
        wetness, _, conf, label = self._probe_scores(emb, self.road_probe)
        return {"label": label.upper(), "wetness": round(wetness, 1),
                "confidence": round(conf, 3)}

    def score_with_probe(self, image) -> tuple[float, np.ndarray, float, str]:
        """Embed-and-score convenience wrapper (kept for the CLI tools)."""
        return self.probe_from(self.embedding(image))

    # ----------------------------------------------------------------------
    def score_context(self, full_image, emb: np.ndarray | None = None) -> dict:
        """Read the SCENE, not the surface. Pass the whole frame, not the ROI.

        Hard shadows mean sun; sun does not coexist with rain. Umbrellas mean
        rain. None of that is visible inside a crop of asphalt, which is what
        makes this independent of the surface classifier rather than another
        way of reading the same pixels.

        `emb` lets a caller that already embedded the full frame (auto-ROI
        shot-change detection does) share the work instead of paying twice.
        """
        if self.context_embeddings is None:
            return {}

        if emb is None:
            emb = self.embedding(full_image)
        sims = emb @ self.context_embeddings.T
        z = sims / CLIP_TEMPERATURE
        z = z - z.max()
        probs = np.exp(z)
        probs /= probs.sum()

        values = np.array([CONTEXT_VALUES[c] for c in self.context_names])
        idx = int(probs.argmax())
        return {
            "scene": self.context_names[idx],
            "scene_confidence": round(float(probs[idx]), 3),
            "rain_likelihood": round(float(probs @ values), 1),
            "probabilities": {
                n: round(float(p), 3)
                for n, p in zip(self.context_names, probs)
            },
        }

    # ----------------------------------------------------------------------
    def embedding(self, image) -> np.ndarray:
        """L2-normalised 512-d CLIP image embedding.

        The raw feature vector, before any prompt comparison. This is what a
        linear probe trains on - prompts are one way to read these features,
        a fitted classifier is another.
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            emb = _as_embedding(self.model.get_image_features(**inputs),
                                self.model, "image")
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb.squeeze(0).numpy()

    def similarities(self, image) -> np.ndarray:
        """Raw cosine similarities against each prompt, before softmax.

        Calibration works from these. Temperature only affects the softmax,
        so sweeping it means re-running four multiplications rather than the
        image encoder - the difference between a second and several minutes
        over a whole calibration set.
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            emb = _as_embedding(self.model.get_image_features(**inputs),
                                self.model, "image")
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return (emb @ self.text_embeddings.T).squeeze(0).numpy()

    def wetness_from_similarities(self, sims: np.ndarray,
                                  temperature: float) -> tuple[float, float]:
        """Turn cached similarities into (wetness, confidence)."""
        z = sims / temperature
        z = z - z.max()                      # stabilise before exp
        probs = np.exp(z)
        probs /= probs.sum()
        return float(probs @ self.anchor_values), float(probs.max())

    # ----------------------------------------------------------------------
    def score(self, image) -> tuple[float, np.ndarray, float]:
        """Return (wetness 0-100, per-prompt probabilities, confidence).

        `image` is a PIL Image, ideally already masked to road pixels.
        """
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            emb = _as_embedding(self.model.get_image_features(**inputs),
                                self.model, "image")
            emb = emb / emb.norm(dim=-1, keepdim=True)

            if emb.shape[-1] != self.text_embeddings.shape[-1]:
                raise RuntimeError(
                    f"embedding dim mismatch: image {emb.shape[-1]} vs text "
                    f"{self.text_embeddings.shape[-1]}")

            sims = (emb @ self.text_embeddings.T).squeeze(0)

        # Measured similarity spread between prompts is only ~0.04, so
        # temperature governs how much of that gap becomes confidence.
        # See CLIP_TEMPERATURE in config - it is a primary tuning knob.
        probs = torch.softmax(sims / CLIP_TEMPERATURE, dim=-1).numpy()
        wetness = float(probs @ self.anchor_values)

        # Confidence = how decisively one prompt won. A flat distribution
        # means CLIP could not discriminate, and the caller should say so
        # rather than assert a state.
        confidence = float(probs.max())

        return wetness, probs, confidence