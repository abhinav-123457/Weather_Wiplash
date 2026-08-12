"""
phrasing.py - a Hugging Face text model writes the suggestion sentence.

THE DIVISION OF LABOUR, AND WHY
-------------------------------
The DECISION is made by rules and stays deterministic. Whether slicks in
standing water is safe is not a matter of opinion, and a generated answer
cannot be safety-checked after the fact - by the time you can read it, it
has already been shown to the user.

The PHRASING is generated. Turning structured facts into a natural radio
message is exactly what a small instruction-tuned model is good at, and
nothing it produces can change what the system decided.

So the model can make the message read better. It cannot make it wrong.

GUARDS
  greedy decoding      no sampling, so the same facts always produce the
                       same sentence - a demo that answers differently on
                       the second run is worse than a plain template
  contradiction check  output naming a tyre the decision did not choose is
                       discarded
  length check         empty, truncated or rambling output is discarded
  template fallback    any failure returns the deterministic sentence, so
                       the feature can never take the system down
"""

from __future__ import annotations

import re

from ..config import PHRASING_MAX_CHARS, PHRASING_MIN_CHARS, PHRASING_MODEL

_TYRE_WORDS = {
    "SLICK": ("slick", "dry tyre", "dry tire"),
    "INTER": ("intermediate", "inter"),
    "FULL_WET": ("full wet", "wet tyre", "wet tire", "extreme"),
}


class Phraser:
    """Rewrites a decided suggestion as a radio message. Optional."""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            from transformers import (AutoModelForSeq2SeqLM,
                                      AutoTokenizer)
            self.tokenizer = AutoTokenizer.from_pretrained(PHRASING_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(PHRASING_MODEL)
            self.model.eval()
            print(f"  phrasing model ready ({PHRASING_MODEL})")
        except Exception as exc:
            print(f"  phrasing model unavailable ({type(exc).__name__}) - "
                  f"templates will be used")

    # ------------------------------------------------------------------
    def _valid(self, text: str, suggested_tire: str,
               current_tire: str) -> bool:
        """Reject anything that contradicts the decision or reads badly."""
        t = text.strip().lower()
        if not (PHRASING_MIN_CHARS <= len(t) <= PHRASING_MAX_CHARS):
            return False
        if t.endswith((",", "and", "the", "to", "for")):
            return False

        # Prompt echo - a leading list dash or a copied field label means
        # the model reflected the input instead of answering it.
        if t.startswith("-") or "facts:" in t or "message:" in t:
            return False

        # Degenerate repetition - flan-t5-small's characteristic failure,
        # observed live: "The weather is a bit dry and wet. - The weather
        # is a bit dry and wet. - ..." three times. The same clause twice
        # is an automatic reject.
        chunks = [c.strip(" .!?-") for c in re.split(r"[.!?]|\s-\s", t)]
        chunks = [c for c in chunks if len(c) > 3]
        if len(chunks) != len(set(chunks)):
            return False

        # Naming a tyre the decision did not choose is the one failure that
        # actually matters: the driver would act on the words, not the JSON.
        for family, words in _TYRE_WORDS.items():
            if family == suggested_tire:
                continue
            if any(w in t for w in words):
                return False

        # Saying nothing actionable is also a failure. A message asking for
        # a tyre change must name the tyre it wants - the observed junk
        # above also never mentioned intermediates at all, and this check
        # alone would have caught it.
        if suggested_tire != current_tire:
            words = _TYRE_WORDS.get(suggested_tire, ())
            if words and not any(w in t for w in words):
                return False
        return True

    # ------------------------------------------------------------------
    def phrase(self, *, headline: str, detail: str | None, urgency: str,
               label: str, trend: str, wetness: float,
               current_tire: str, suggested_tire: str,
               laps: int | None) -> dict:
        """Return {"text", "source"} - source is 'model' or 'template'."""
        fallback = headline if not detail else f"{headline} — {detail}"

        if self.model is None:
            return {"text": fallback, "source": "template"}

        facts = [
            f"track condition is {label.lower()}",
            f"wetness {wetness:.0f} out of 100",
            f"trend is {trend.lower()}",
            f"car is on {current_tire.replace('_', ' ').lower()} tyres",
        ]
        if suggested_tire != current_tire:
            facts.append(f"should change to "
                         f"{suggested_tire.replace('_', ' ').lower()} tyres")
        else:
            facts.append("no tyre change needed")
        if laps:
            facts.append(f"slick tyres become faster in about {laps} laps")
        if urgency == "URGENT":
            facts.append("this is urgent")

        prompt = (
            "Write one short radio message from a race engineer to a Formula 1 "
            "driver about the track conditions. Be direct and under 20 words.\n"
            f"Facts: {'; '.join(facts)}.\n"
            "Message:"
        )

        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt",
                                    truncation=True, max_length=256)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False,       # deterministic - see module docstring
                    num_beams=2,
                )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        except Exception:
            return {"text": fallback, "source": "template"}

        if not self._valid(text, suggested_tire, current_tire):
            return {"text": fallback, "source": "template",
                    "rejected": text[:120]}

        return {"text": text, "source": "model"}
