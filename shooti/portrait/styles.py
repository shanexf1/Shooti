"""Portrait styles, as tolerance overrides.

A formal headshot and a documentary portrait are judged differently by any real
photographer: the headshot wants a level head, a clean background and a sharp
eye; the documentary frame tolerates a cant, a busy street behind, and a
background as sharp as the face because that background is the point.

v4 applied one set of tolerances to every portrait. These let the judging adapt.
NEUTRAL reproduces v4's constants exactly, so v4 and v4.1 are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    key: str
    label: str
    applies_when: str

    tilt_excessive: float = 22.0
    pitch_up_limit: float = -9.0
    pitch_down_limit: float = 16.0
    nose_room_min: float = 0.55
    eye_focus_min: float = 1.15
    clutter_high: float = 0.14
    blur_wanted: float = 1.5
    separation_low: float = 0.10
    crop_joint_severity: str = "major"  # documentary work crops through joints on purpose


NEUTRAL = Style(
    key="neutral",
    label="No style stated",
    applies_when="Use when the photographer has given no indication of style.",
)

FORMAL = Style(
    key="formal",
    label="Formal headshot",
    applies_when="Corporate, LinkedIn, passport, yearbook, actor's headshot. Clean and neutral.",
    tilt_excessive=12.0,       # a cant reads as sloppy here
    pitch_up_limit=-6.0,
    pitch_down_limit=12.0,
    eye_focus_min=1.4,         # a soft eye is disqualifying
    clutter_high=0.08,         # background should be plain
    blur_wanted=2.0,
    separation_low=0.14,
)

EDITORIAL = Style(
    key="editorial",
    label="Editorial / portrait with intent",
    applies_when="Magazine, band, author or campaign portrait where a strong look is wanted.",
    tilt_excessive=28.0,
    clutter_high=0.18,
    blur_wanted=1.3,
)

ENVIRONMENTAL = Style(
    key="environmental",
    label="Environmental portrait",
    applies_when=(
        "The subject shown in their place — workshop, kitchen, field, studio — where the "
        "surroundings are part of the story."
    ),
    clutter_high=0.30,         # the environment is the point
    blur_wanted=0.9,           # a readable background is wanted
    separation_low=0.07,
    nose_room_min=0.45,
)

CANDID = Style(
    key="candid",
    label="Candid / documentary",
    applies_when="Unposed, reportage, street, family life. Timing outranks tidiness.",
    tilt_excessive=32.0,
    pitch_up_limit=-16.0,
    pitch_down_limit=26.0,
    nose_room_min=0.40,
    eye_focus_min=0.95,
    clutter_high=0.34,
    blur_wanted=0.8,
    separation_low=0.06,
    crop_joint_severity="minor",
)

BEAUTY = Style(
    key="beauty",
    label="Beauty / close-up",
    applies_when="Tight beauty, makeup or skin-detail work, usually cropping the crown.",
    tilt_excessive=26.0,
    eye_focus_min=1.6,
    clutter_high=0.06,
    blur_wanted=2.2,
    separation_low=0.12,
)

STYLES: dict[str, Style] = {
    s.key: s for s in (NEUTRAL, FORMAL, EDITORIAL, ENVIRONMENTAL, CANDID, BEAUTY)
}

# Keyword fallback, for when no LLM key is present.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "formal": ("headshot", "corporate", "linkedin", "professional", "passport",
               "yearbook", "business", "id photo", "actor"),
    "editorial": ("editorial", "magazine", "campaign", "band", "author", "fashion",
                  "moody", "dramatic", "cinematic"),
    "environmental": ("environmental", "at work", "workshop", "kitchen", "studio",
                      "in their", "on location", "workplace", "farm", "office"),
    "candid": ("candid", "documentary", "street", "unposed", "reportage", "family",
               "everyday", "playing", "spontaneous"),
    "beauty": ("beauty", "makeup", "skin", "close-up", "closeup", "glamour", "lips", "eyes"),
}


def catalog_for_prompt() -> str:
    return "\n".join(
        f"- {s.key}: {s.applies_when}" for s in STYLES.values() if s.key != "neutral"
    ) + "\n- neutral: " + NEUTRAL.applies_when


def from_keywords(text: str) -> tuple[Style, str]:
    """Deterministic fallback. Returns (style, why)."""
    import re

    t = f" {(text or '').lower().strip()} "
    if not t.strip():
        return NEUTRAL, "No style described, so v4's default tolerances are used."
    scores: dict[str, float] = {}
    hits: dict[str, list[str]] = {}
    for key, words in KEYWORDS.items():
        for w in words:
            if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", t):
                scores[key] = scores.get(key, 0.0) + 1.0 + 0.5 * w.count(" ")
                hits.setdefault(key, []).append(w)
    if not scores:
        return NEUTRAL, "No recognised style keywords, so v4's default tolerances are used."
    best = max(scores, key=lambda k: scores[k])
    return STYLES[best], f"Matched on: {', '.join(sorted(set(hits[best])))} (keyword fallback)."
