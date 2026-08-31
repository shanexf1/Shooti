"""Turn measurements into photographer-readable coaching, via Claude vision.

The CV layer already knows the geometry. Claude's job is the part geometry can't
do: reading the scene (what the subject is, where the light is, what the
background is doing) and prioritizing the moves into advice a beginner can act
on while still holding the camera.

The measurements are passed in as text alongside the image so the model grounds
its advice in real numbers instead of guessing at pixel positions.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import anthropic
import cv2
import numpy as np

from .rules import Analysis

MODEL = "claude-opus-5"
MAX_EDGE = 1024  # downscale before upload; framing advice doesn't need full res

SYSTEM = """You are a photography coach standing next to a beginner who is \
about to take this shot. You are given the photo plus geometric measurements \
from a computer-vision pass.

Rules:
- Trust the measurements for geometry (position, tilt, headroom, size). Do not \
contradict them or re-estimate numbers yourself.
- Add what the measurements cannot see: what the subject actually is, the \
background, the light, the moment.
- Give at most three moves, ordered by how much they improve the shot. Each \
move must be something the photographer can physically do right now: step, \
crouch, pan, tilt, rotate, wait, or reframe.
- Then one short line naming what already works, so they keep it.
- Plain language. No jargon without a gloss. No preamble, no sign-off.

Format exactly:
MOVES
1. <move> — <why it helps>
2. ...
KEEP
<one sentence>"""


@dataclass
class Coaching:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class CoachError(RuntimeError):
    pass


def _encode(bgr: np.ndarray) -> str:
    h, w = bgr.shape[:2]
    scale = MAX_EDGE / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise CoachError("Could not JPEG-encode the frame.")
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def measurements_text(analysis: Analysis) -> str:
    """Flatten the analysis into the grounding block Claude reads."""
    s = analysis.subject
    ax, ay = analysis.anchor
    lines = [
        f"frame: {analysis.width}x{analysis.height}",
        f"subject: {s.kind} (confidence {s.confidence:.2f}), "
        f"filling {s.area_fraction * 100:.0f}% of the frame",
        f"anchor point ({'eye level' if s.face else 'subject center'}): "
        f"{ax / analysis.width:.2f}, {ay / analysis.height:.2f} of frame",
        f"thirds target for that anchor: "
        f"{analysis.target[0] / analysis.width:.2f}, {analysis.target[1] / analysis.height:.2f}",
    ]
    if s.face is not None:
        yaw = s.face.yaw
        turned = "facing camera" if abs(yaw) < 0.15 else f"turned toward frame {'right' if yaw > 0 else 'left'}"
        lines.append(f"faces detected: {s.face_count}; primary head {turned} (yaw index {yaw:+.2f})")
    if analysis.horizon:
        lines.append(
            f"horizon: {analysis.horizon.angle_deg:+.1f} deg from level, "
            f"crossing frame center at y={analysis.horizon.y / analysis.height:.2f}, "
            f"detection strength {analysis.horizon.strength:.2f}"
        )
    else:
        lines.append("horizon: none detected")
    lines.append(f"rule-based composition score: {analysis.score}/100")
    lines.append("findings:")
    for f in analysis.findings:
        lines.append(f"  [{f.severity}] {f.rule}: {f.message} -> {f.action}")
    return "\n".join(lines)


def coach(
    bgr: np.ndarray,
    analysis: Analysis,
    *,
    api_key: str | None = None,
    intent: str | None = None,
) -> Coaching:
    """Ask Claude for prioritized moves. Raises CoachError on failure."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    # A bare client also resolves an `ant auth login` profile, so an unset env
    # var is not automatically an error — let the SDK try.
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    prompt = measurements_text(analysis)
    if intent:
        prompt += f"\n\nphotographer's intent: {intent}"

    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=2048,  # the output format is deliberately four short lines
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},  # interactive path — latency matters
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
        # The SDK raises a bare TypeError -- not AuthenticationError -- when no
        # credential can be resolved, so this needs its own arm or the UI shows
        # a raw "TypeError" to the user.
        raise CoachError("No Anthropic credentials found. Set ANTHROPIC_API_KEY, paste a key in the sidebar, or run `ant auth login`.") from exc
    except anthropic.AuthenticationError as exc:
        raise CoachError(
            "Claude rejected the credentials. Set ANTHROPIC_API_KEY or run `ant auth login`."
        ) from exc
    except anthropic.RateLimitError as exc:
        retry = exc.response.headers.get("retry-after", "60")
        raise CoachError(f"Rate limited by the API. Try again in {retry}s.") from exc
    except anthropic.APIStatusError as exc:
        raise CoachError(f"Claude API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise CoachError("Could not reach the Claude API. Check your connection.") from exc

    if response.stop_reason == "refusal":
        detail = response.stop_details.explanation if response.stop_details else ""
        raise CoachError(f"Claude declined to analyze this image. {detail}".strip())

    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise CoachError("Claude returned no text.")

    return Coaching(
        text=text,
        model=response.model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
