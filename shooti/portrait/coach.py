"""Portrait coaching from either Claude or ChatGPT, switchable at runtime.

The measurements stay in Python. The model is given the photo plus the numbers
v4 already computed and told not to re-estimate geometry — same division of
labour as v1's coach, for the same reason: a language model asked to judge
headroom will invent a percentage, while the CV layer can measure it.

Both providers get an identical system prompt and identical grounding text, so
switching providers compares the models rather than two different prompts.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .rules import PortraitVerdict

MAX_EDGE = 1024

DEFAULT_MODELS = {
    "claude": "claude-opus-5",
    "openai": "gpt-4o",
}

PROVIDER_LABELS = {
    "claude": "Claude (Anthropic)",
    "openai": "ChatGPT (OpenAI)",
}

KEY_ENV = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

SYSTEM = """You are a portrait photographer coaching someone on a specific shot.

You get the photograph plus measurements from a computer-vision pass: the crop \
classified in head-heights, the eye line, headroom, head pose in degrees \
relative to the lens, eye sharpness, face exposure, and what is behind the head.

Rules:
- Trust the measurements for geometry and exposure. Do not restate them as your \
own estimates and do not contradict them.
- The measurements assume an upright adult subject. If the photo shows someone \
seated, reclining, or a child, say so — it invalidates the crop-line finding.
- Add only what measurement cannot see: expression, gesture, wardrobe, styling, \
where the subject's attention is, whether the moment works.
- Give at most three actions, hardest-hitting first. Each must be something the \
photographer can do on the next frame: move, change height, redirect the \
subject, change a setting, wait.
- Then one line on what already works.
- Speak plainly to the photographer. No preamble, no sign-off, no headings \
beyond the two below.

Format exactly:
DO THIS
1. <action> — <why>
2. ...
KEEP
<one sentence>"""


@dataclass
class Coaching:
    text: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


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


def measurements_text(verdict: PortraitVerdict, intent: str | None = None) -> str:
    """The grounding block. Identical for both providers."""
    c, p = verdict.crop, verdict.pose
    lines = [
        f"crop: {c.crop_name}; frame ends at {c.heads_to_bottom:.2f} head-heights from the crown",
        f"joint cut by the frame: {c.joint or 'none'}",
        f"head height (estimated): {c.head_height_px:.0f}px",
    ]
    if p.ok:
        lines.append(
            f"head pose vs lens: yaw {p.yaw_deg:+.1f} deg (+ = turned toward frame right), "
            f"pitch {p.pitch_deg:+.1f} deg (+ = face angled down), roll {p.roll_deg:+.1f} deg"
        )
    else:
        lines.append(f"head pose: unavailable ({p.note})")
    lines.append(f"human-subject check: {verdict.human.verdict}")
    lines.append(f"portrait rule score: {verdict.score}/100")
    lines.append("findings:")
    for f in verdict.findings:
        lines.append(f"  [{f.severity}] {f.rule}: {f.message} -> {f.action}")
    if verdict.notes:
        lines.append("caveats the analysis itself flagged:")
        for n in verdict.notes:
            lines.append(f"  - {n}")
    if intent:
        lines.append(f"photographer's intent: {intent}")
    return "\n".join(lines)


NO_KEY = "No {name} key found. Paste one in the sidebar or set {env}."


def _claude(prompt: str, image_b64: str, model: str, api_key: str | None) -> Coaching:
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=2048,  # output is a fixed short format
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
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
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except TypeError as exc:
        # The SDK raises a bare TypeError when no credential resolves.
        raise CoachError(NO_KEY.format(name="Anthropic", env="ANTHROPIC_API_KEY")) from exc
    except anthropic.AuthenticationError as exc:
        raise CoachError("Anthropic rejected that key.") from exc
    except anthropic.NotFoundError as exc:
        raise CoachError(f"Anthropic has no model {model!r}. Change the model field.") from exc
    except anthropic.RateLimitError as exc:
        raise CoachError("Rate limited by Anthropic. Try again shortly.") from exc
    except anthropic.APIStatusError as exc:
        raise CoachError(f"Anthropic error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise CoachError("Could not reach Anthropic. Check your connection.") from exc

    if response.stop_reason == "refusal":
        raise CoachError("Claude declined to analyze this image.")
    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        raise CoachError("Claude returned no text.")
    return Coaching(
        text, "claude", response.model,
        response.usage.input_tokens, response.usage.output_tokens,
    )


def _openai(prompt: str, image_b64: str, model: str, api_key: str | None) -> Coaching:
    import openai

    key = api_key or os.environ.get("OPENAI_API_KEY")
    try:
        client = openai.OpenAI(api_key=key) if key else openai.OpenAI()
    except openai.OpenAIError as exc:
        raise CoachError(NO_KEY.format(name="OpenAI", env="OPENAI_API_KEY")) from exc

    try:
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )
    except openai.AuthenticationError as exc:
        raise CoachError("OpenAI rejected that key.") from exc
    except openai.NotFoundError as exc:
        raise CoachError(
            f"OpenAI has no model {model!r} available to this key. Use the "
            "'List my models' button to see what you can call."
        ) from exc
    except openai.PermissionDeniedError as exc:
        raise CoachError(f"This key may not use {model!r}.") from exc
    except openai.RateLimitError as exc:
        raise CoachError("Rate limited by OpenAI (or the account is out of quota).") from exc
    except openai.APIStatusError as exc:
        raise CoachError(f"OpenAI error {exc.status_code}: {exc.message}") from exc
    except openai.APIConnectionError as exc:
        raise CoachError("Could not reach OpenAI. Check your connection.") from exc

    choice = response.choices[0] if response.choices else None
    text = (choice.message.content or "").strip() if choice else ""
    if not text:
        reason = getattr(choice, "finish_reason", "unknown") if choice else "no choices"
        raise CoachError(f"OpenAI returned no text (finish_reason={reason}).")
    usage = response.usage
    return Coaching(
        text, "openai", response.model,
        getattr(usage, "prompt_tokens", None) if usage else None,
        getattr(usage, "completion_tokens", None) if usage else None,
    )


def coach(
    bgr: np.ndarray,
    verdict: PortraitVerdict,
    *,
    provider: str = "claude",
    api_key: str | None = None,
    model: str | None = None,
    intent: str | None = None,
) -> Coaching:
    if provider not in DEFAULT_MODELS:
        raise CoachError(f"Unknown provider {provider!r}.")
    model = model or DEFAULT_MODELS[provider]
    prompt = measurements_text(verdict, intent)
    image = _encode(bgr)
    backend = _claude if provider == "claude" else _openai
    return backend(prompt, image, model, api_key)


def list_models(provider: str, api_key: str | None = None) -> list[str]:
    """What can this key actually call? Saves guessing at model names."""
    try:
        if provider == "openai":
            import openai

            key = api_key or os.environ.get("OPENAI_API_KEY")
            client = openai.OpenAI(api_key=key) if key else openai.OpenAI()
            return sorted(m.id for m in client.models.list())
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        return sorted(m.id for m in client.models.list())
    except Exception as exc:
        raise CoachError(f"Could not list models: {type(exc).__name__}: {exc}") from exc
