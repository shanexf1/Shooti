"""Pick the rule set for a photo, given the photographer's stated intent.

Two paths, and the app is honest about which one ran:

  claude  — sends the photo plus the intent text and lets the model choose from
            the profile catalog. Handles free-form intent ("I want this to feel
            lonely") that keyword matching cannot.
  keyword — a deterministic backstop so the app works with no API key at all.

The LLM's answer is validated against the known profile keys. A model that
invents a key does not get to pick: selection falls back and says so.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

from .profiles import GENERIC, KEYWORDS, PROFILES, Profile, catalog_for_prompt

MODEL = "claude-opus-5"
MAX_EDGE = 768  # the choice needs the gist of the frame, not detail


@dataclass
class Selection:
    profile: Profile
    source: str  # "claude" | "keyword" | "default" | "manual"
    reasoning: str
    runner_up: Profile | None = None
    note: str | None = None  # e.g. why a fallback happened


class SelectError(RuntimeError):
    pass


# ---------------------------------------------------------------- keyword path


def select_keyword(intent: str) -> Selection:
    """Count keyword hits per profile. Longest phrases win ties by being specific."""
    text = f" {intent.lower().strip()} "
    if not text.strip():
        return Selection(GENERIC, "default", "No intent given, so no rule set can be chosen for it.")

    scores: dict[str, float] = {}
    hits: dict[str, list[str]] = {}
    for key, words in KEYWORDS.items():
        for w in words:
            if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text):
                # Multi-word phrases are stronger evidence than single words.
                scores[key] = scores.get(key, 0.0) + 1.0 + 0.5 * w.count(" ")
                hits.setdefault(key, []).append(w)

    if not scores:
        return Selection(
            GENERIC, "default",
            "No recognized intent keywords, so the universal rule set is used — "
            "which has no measured signal. Naming the intent would help.",
        )

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best = PROFILES[ranked[0][0]]
    runner = PROFILES[ranked[1][0]] if len(ranked) > 1 else None
    matched = ", ".join(sorted(set(hits[ranked[0][0]])))
    return Selection(
        best, "keyword",
        f"Matched on: {matched}. Keyword fallback — no LLM was consulted.",
        runner_up=runner,
    )


# ------------------------------------------------------------------ claude path

SYSTEM = """You choose which composition rule set applies to a photograph.

You are given the photo and the photographer's stated intent. Pick exactly one \
rule set from the catalog. Some rule sets deliberately switch rules off — pick \
the one whose assumptions match the photographer's goal, not the one that scores \
most rules.

The photographer's stated intent is the primary signal. Use the photo to resolve \
ambiguity and to catch a mismatch: if the stated intent and the image plainly \
disagree, choose for the intent and say so in your reason.

Respond in exactly this format and nothing else:

PROFILE: <key>
RUNNER_UP: <key or none>
REASON: <one sentence, addressed to the photographer>"""


def _encode(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    scale = MAX_EDGE / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise SelectError("Could not JPEG-encode the frame.")
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def _parse(text: str) -> tuple[str | None, str | None, str]:
    profile = runner = None
    reason = ""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("PROFILE:"):
            profile = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("RUNNER_UP:"):
            val = line.split(":", 1)[1].strip().lower()
            runner = None if val in ("none", "", "n/a") else val
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return profile, runner, reason


def select_claude(bgr: np.ndarray, intent: str, api_key: str | None = None) -> Selection:
    """Ask Claude to choose. Raises SelectError so the caller can fall back."""
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    prompt = (
        f"Photographer's stated intent:\n{intent.strip() or '(none given)'}\n\n"
        f"Available rule sets:\n{catalog_for_prompt()}"
    )

    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=1024,  # the reply is three short lines by construction
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},  # a menu choice, not a hard problem
            system=SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": _encode(bgr),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except TypeError as exc:
        # The SDK raises a bare TypeError -- not AuthenticationError -- when it
        # cannot resolve any credential at request time.
        raise SelectError("No Anthropic credentials found. Set ANTHROPIC_API_KEY, paste a key in the sidebar, or run `ant auth login`.") from exc
    except anthropic.AuthenticationError as exc:
        raise SelectError("Claude rejected the credentials.") from exc
    except anthropic.RateLimitError as exc:
        raise SelectError("Rate limited by the Claude API.") from exc
    except anthropic.APIStatusError as exc:
        raise SelectError(f"Claude API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise SelectError("Could not reach the Claude API.") from exc

    if response.stop_reason == "refusal":
        raise SelectError("Claude declined to analyze this image.")

    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    key_out, runner_out, reason = _parse(text)

    if key_out not in PROFILES:
        raise SelectError(
            f"Claude returned an unknown rule set {key_out!r}. Not trusting it."
        )
    return Selection(
        PROFILES[key_out],
        "claude",
        reason or "(no reason given)",
        runner_up=PROFILES.get(runner_out) if runner_out else None,
    )


# ----------------------------------------------------------------- entry point


def select(
    bgr: np.ndarray,
    intent: str,
    *,
    use_llm: bool = True,
    api_key: str | None = None,
) -> Selection:
    """Claude if available, keyword otherwise. Always reports which one ran."""
    if use_llm:
        try:
            return select_claude(bgr, intent, api_key=api_key)
        except SelectError as exc:
            fallback = select_keyword(intent)
            fallback.note = f"Claude unavailable ({exc}). Fell back to keyword matching."
            return fallback
        except Exception as exc:  # never let selection break the whole page
            fallback = select_keyword(intent)
            fallback.note = f"Claude call failed ({type(exc).__name__}). Fell back to keywords."
            return fallback
    return select_keyword(intent)
