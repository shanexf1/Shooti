"""Let the model adjust the judging, with guardrails and a full audit trail.

Until now the LLM was downstream of judging: it read the findings and narrated
them. It could not change the verdict. That is a real limitation, because the
measurements have known blind spots the model can see straight past:

  - "vertical intrusion" finds a LINE, not a pole. It cannot tell a lamp post
    behind the head from the subject's own raised arm.
  - the crop-line rule assumes an upright adult, and cannot see that the subject
    is sitting.
  - "colour distraction" cannot tell a red road sign from the subject's red coat.

So the model now does two things that change the verdict:

  1. Picks the STYLE, which shifts every tolerance (a documentary frame is not
     judged like a passport photo).
  2. DISMISSES individual findings it can see are wrong, or ESCALATES ones the
     measurement understated — each with a stated reason.

The obvious risk is a model that dismisses everything to be agreeable. Hence the
guardrails below: a hard cap on dismissals, a required reason for each, a cap on
how much the score may rise, and every decision shown to the user with its
reason. Nothing is silently changed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ..rules import Finding
from .coach import CoachError, _encode, measurements_text
from .styles import NEUTRAL, STYLES, Style, catalog_for_prompt, from_keywords

MAX_DISMISS = 3
MAX_ESCALATE = 3  # symmetric with dismissals; asymmetry favoured leniency
MAX_ADD = 3
MAX_SCORE_GAIN = 25  # dismissals may not lift the score by more than this
MAX_SCORE_LOSS = 40  # escalations + additions may not sink it by more than this
MIN_REASON_CHARS = 15
ADD_PENALTY = {"major": 14.0, "minor": 7.0}
ESCALATE_CAP = 26.0  # one rule cannot dominate the whole score

SYSTEM = """You are reviewing an automated portrait critique before it reaches the \
photographer. You see the photograph, the photographer's stated intent, and every \
finding the computer-vision rules produced.

Your job is to correct the critique where the measurements are wrong about this \
specific photo, and to pick the style whose tolerances fit the photographer's goal.

The measurements have known blind spots. They detect lines, not objects; blobs, \
not meanings. A "vertical intrusion" may be the subject's own arm. A "colour \
distraction" may be their coat. The crop-line rule assumes an upright adult and \
does not know if they are seated. Dismiss a finding ONLY when you can see the \
specific reason it does not apply here, and say what you see.

Do not dismiss a finding merely because it is unwelcome, or to be encouraging. A \
real problem left standing helps the photographer; a dismissed one does not. If \
every finding looks correct, dismiss nothing — that is the expected answer.

You can also ADD a problem the rules have no way to measure. The rules see
geometry, sharpness and tone — they cannot see a closed eye, a forced smile, a
distracting hand, hair across the face, a crooked collar, or a subject who looks
uncomfortable. If something like that is wrong, add it. Adding a real problem is
as useful as dismissing a false one, and you are expected to do both.

Respond in exactly this format and nothing else:

STYLE: <one style key>
STYLE_REASON: <one sentence>
DISMISS: <rule name> | <what you see that makes this finding wrong>
DISMISS: ... (at most %d lines, omit entirely if none)
ESCALATE: <rule name> | <why this is worse than measured>
ESCALATE: ... (at most %d lines, omit entirely if none)
ADD: <short name> | major or minor | <what is wrong> | <what to do about it>
ADD: ... (at most %d lines, omit entirely if none)""" % (MAX_DISMISS, MAX_ESCALATE, MAX_ADD)


@dataclass
class Decision:
    rule: str
    action: str  # "dismiss" | "escalate" | "add"
    reason: str
    applied: bool
    blocked_because: str | None = None


@dataclass
class Adjudication:
    style: Style
    style_reason: str
    decisions: list[Decision] = field(default_factory=list)
    source: str = "keyword"  # "llm" | "keyword"
    raw_score: int = 0  # under NEUTRAL tolerances
    style_score: int = 0  # after re-judging under the chosen style
    final_score: int = 0  # after dismissals and escalations
    note: str | None = None

    @property
    def style_delta(self) -> int:
        return self.style_score - self.raw_score

    @property
    def review_delta(self) -> int:
        return self.final_score - self.style_score

    @property
    def applied(self) -> list[Decision]:
        return [d for d in self.decisions if d.applied]

    @property
    def blocked(self) -> list[Decision]:
        return [d for d in self.decisions if not d.applied]


def _parse(text: str):
    style, reason = "neutral", ""
    dismiss: list[tuple[str, str]] = []
    escalate: list[tuple[str, str]] = []
    added: list[tuple[str, str, str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("STYLE:"):
            style = line.split(":", 1)[1].strip().lower()
        elif upper.startswith("STYLE_REASON:"):
            reason = line.split(":", 1)[1].strip()
        elif upper.startswith("DISMISS:"):
            body = line.split(":", 1)[1]
            if "|" in body:
                r, why = body.split("|", 1)
                dismiss.append((r.strip().lower(), why.strip()))
        elif upper.startswith("ESCALATE:"):
            body = line.split(":", 1)[1]
            if "|" in body:
                r, why = body.split("|", 1)
                escalate.append((r.strip().lower(), why.strip()))
        elif upper.startswith("ADD:"):
            parts = [q.strip() for q in line.split(":", 1)[1].split("|")]
            if len(parts) >= 3:
                name = parts[0][:40]
                sev = parts[1].lower()
                sev = sev if sev in ("major", "minor") else "minor"
                msg = parts[2]
                act = parts[3] if len(parts) > 3 else "Fix this before the next frame."
                added.append((name, sev, msg, act))
    return style, reason, dismiss, escalate, added


def _ask_llm(bgr, verdict, provider, api_key, model, intent):
    prompt = (
        f"Photographer's stated intent:\n{(intent or '(none given)').strip()}\n\n"
        f"Available styles:\n{catalog_for_prompt()}\n\n"
        f"The automated critique:\n{measurements_text(verdict, intent)}"
    )
    image = _encode(bgr)

    if provider == "claude":
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        try:
            resp = client.beta.messages.create(
                model=model or "claude-opus-5",
                max_tokens=2048,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                system=SYSTEM,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg", "data": image}},
                    {"type": "text", "text": prompt},
                ]}],
            )
        except TypeError as exc:
            raise CoachError("No Anthropic key found. Paste one in the sidebar or set "
                             "ANTHROPIC_API_KEY.") from exc
        except anthropic.APIStatusError as exc:
            raise CoachError(f"Anthropic error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise CoachError("Could not reach Anthropic.") from exc
        if resp.stop_reason == "refusal":
            raise CoachError("Claude declined to review this image.")
        text = "\n".join(b.text for b in resp.content if b.type == "text")
    else:
        import openai

        key = api_key or os.environ.get("OPENAI_API_KEY")
        try:
            client = openai.OpenAI(api_key=key) if key else openai.OpenAI()
        except openai.OpenAIError as exc:
            raise CoachError("No OpenAI key found. Paste one in the sidebar or set "
                             "OPENAI_API_KEY.") from exc
        try:
            resp = client.chat.completions.create(
                model=model or "gpt-4o",
                max_completion_tokens=2048,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{image}", "detail": "high"}},
                        {"type": "text", "text": prompt},
                    ]},
                ],
            )
        except openai.APIStatusError as exc:
            raise CoachError(f"OpenAI error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise CoachError("Could not reach OpenAI.") from exc
        text = (resp.choices[0].message.content or "") if resp.choices else ""

    if not text.strip():
        raise CoachError("The model returned no text.")
    return _parse(text)


def adjudicate(
    bgr: np.ndarray,
    face,
    all_faces,
    raw_verdict,
    *,
    use_llm: bool = True,
    provider: str = "claude",
    api_key: str | None = None,
    model: str | None = None,
    intent: str | None = None,
    deep_background: bool = True,
):
    """Return (verdict, Adjudication). Falls back to keywords with no LLM."""
    from .rules import analyze_portrait

    raw_score = raw_verdict.score
    style, style_reason = NEUTRAL, ""
    dismiss: list[tuple[str, str]] = []
    escalate: list[tuple[str, str]] = []
    added: list[tuple[str, str, str, str]] = []
    source, note = "keyword", None

    if use_llm:
        try:
            key, style_reason, dismiss, escalate, added = _ask_llm(
                bgr, raw_verdict, provider, api_key, model, intent
            )
            style = STYLES.get(key, NEUTRAL)
            if key not in STYLES:
                style_reason = f"(model returned unknown style {key!r}; using neutral) " + style_reason
            source = "llm"
        except CoachError as exc:
            style, style_reason = from_keywords(intent or "")
            note = f"{exc} Fell back to keyword style selection, and no findings were reviewed."
        except Exception as exc:
            style, style_reason = from_keywords(intent or "")
            note = f"Review failed ({type(exc).__name__}). Keyword style only."
    else:
        style, style_reason = from_keywords(intent or "")

    # Re-judge under the chosen style.
    verdict = analyze_portrait(
        bgr, face, all_faces=all_faces, deep_background=deep_background, style=style
    )

    # Two separate effects, reported separately: re-judging under a different
    # style, and the model's per-finding decisions. Conflating them hides which
    # one moved the score.
    style_score = verdict.score

    by_rule = {f.rule.lower(): f for f in verdict.findings}
    decisions: list[Decision] = []
    gained = 0.0

    for rule, reason in dismiss[:MAX_DISMISS]:
        target = by_rule.get(rule)
        if target is None:
            decisions.append(Decision(rule, "dismiss", reason, False,
                                      "no such finding in this critique"))
            continue
        if target.severity == "ok":
            decisions.append(Decision(rule, "dismiss", reason, False,
                                      "that finding already passed"))
            continue
        if len(reason) < MIN_REASON_CHARS:
            decisions.append(Decision(rule, "dismiss", reason, False,
                                      "no specific reason given"))
            continue
        if gained + target.penalty > MAX_SCORE_GAIN:
            decisions.append(Decision(rule, "dismiss", reason, False,
                                      f"would lift the score past the +{MAX_SCORE_GAIN} cap"))
            continue
        gained += target.penalty
        target.data["dismissed_reason"] = reason
        target.severity = "ok"
        target.message = f"[dismissed on review] {target.message}"
        target.penalty = 0.0
        decisions.append(Decision(rule, "dismiss", reason, True))

    lost = 0.0
    for rule, reason in escalate[:MAX_ESCALATE]:
        target = by_rule.get(rule)
        if target is None:
            decisions.append(Decision(rule, "escalate", reason, False,
                                      "no such finding in this critique"))
            continue
        if len(reason) < MIN_REASON_CHARS:
            decisions.append(Decision(rule, "escalate", reason, False,
                                      "no specific reason given"))
            continue
        # Escalation must actually cost something. Flooring at 12 was a no-op on
        # any finding already penalised above that, which reported APPLIED while
        # moving the score by zero.
        new_penalty = min(max(target.penalty * 1.5, 12.0), ESCALATE_CAP)
        bump = new_penalty - target.penalty
        if bump <= 0.01:
            decisions.append(Decision(rule, "escalate", reason, False,
                                      "already penalised at the maximum for this rule"))
            continue
        if lost + bump > MAX_SCORE_LOSS:
            decisions.append(Decision(rule, "escalate", reason, False,
                                      f"would sink the score past the -{MAX_SCORE_LOSS} cap"))
            continue
        lost += bump
        target.severity = "major"
        target.penalty = new_penalty
        target.data["escalated_reason"] = reason
        decisions.append(Decision(rule, "escalate", reason, True))

    # Problems the rules cannot measure at all — a closed eye, an awkward hand,
    # hair across the face. Without this the model can only ever be lenient.
    for name, sev, msg, act in added[:MAX_ADD]:
        if len(msg) < MIN_REASON_CHARS:
            decisions.append(Decision(name, "add", msg, False, "no specific problem described"))
            continue
        penalty = ADD_PENALTY[sev]
        if lost + penalty > MAX_SCORE_LOSS:
            decisions.append(Decision(name, "add", msg, False,
                                      f"would sink the score past the -{MAX_SCORE_LOSS} cap"))
            continue
        lost += penalty
        verdict.findings.append(Finding(
            name, sev, msg, act, penalty=penalty,
            data={"added_by_model": True},
        ))
        decisions.append(Decision(name, "add", msg, True))

    verdict.score = int(round(max(0.0, 100.0 - sum(f.penalty for f in verdict.findings))))
    order = {"major": 0, "minor": 1, "ok": 2}
    verdict.findings.sort(key=lambda f: (order[f.severity], -f.penalty))

    return verdict, Adjudication(
        style=style, style_reason=style_reason, decisions=decisions,
        source=source, raw_score=raw_score, style_score=style_score,
        final_score=verdict.score, note=note,
    )
