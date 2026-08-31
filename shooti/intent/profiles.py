"""Rule sets, one per photographic intent.

v1's mistake was a single rule set applied to every photograph: it penalized a
dead-center subject even when centering was the whole point. Measured against
human ratings, that universal score carried no signal (SRCC -0.006).

The hypothesis here is narrower and testable: rules work when the *right* rules
are applied. So each intent gets its own set — a symmetric architecture shot
should be rewarded for centering, not penalized; a street photograph should not
be scolded for a deliberate cant.

A tolerance of None means the rule DOES NOT APPLY to this intent, and the engine
skips it entirely rather than scoring it leniently. That distinction matters:
"not applicable" and "barely penalized" say different things to a photographer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    blurb: str  # shown in the UI
    applies_when: str  # shown to the LLM when it picks

    # None = rule not applicable for this intent.
    thirds_tol: float | None = 0.055
    tilt_tol_deg: float | None = 1.5
    headroom: tuple[float, float] | None = (0.03, 0.14)
    pitch_tol: float | None = 0.06
    edge_tol: float | None = 0.008
    size_range: tuple[float, float] | None = (0.06, 0.62)
    balance_tol: float | None = 0.22
    yaw_tol: float | None = 0.15

    # Inversions — where an intent reverses v1's assumption outright.
    center_is_good: bool = False  # reward centering instead of penalizing it
    symmetry_matters: bool = False  # penalize left/right imbalance hard

    # Per-rule penalty multipliers. Absent = 1.0.
    weights: dict[str, float] = field(default_factory=dict)

    def applies(self, rule: str) -> bool:
        return {
            "thirds": self.thirds_tol,
            "tilt": self.tilt_tol_deg,
            "headroom": self.headroom,
            "camera pitch": self.pitch_tol,
            "edges": self.edge_tol,
            "subject size": self.size_range,
            "balance": self.balance_tol,
            "looking room": self.yaw_tol,
        }.get(rule) is not None

    def weight(self, rule: str) -> float:
        return float(self.weights.get(rule, 1.0))


PORTRAIT = Profile(
    key="portrait",
    label="Portrait",
    blurb="A person is the subject. Eye level and headroom carry the frame.",
    applies_when="A posed or candid photograph of one person where the face is the subject.",
    thirds_tol=0.06,
    headroom=(0.03, 0.13),
    tilt_tol_deg=2.5,  # a slight cant is a common portrait choice
    size_range=(0.08, 0.70),
    weights={"headroom": 1.4, "looking room": 1.5, "thirds": 1.1, "balance": 0.5},
)

GROUP = Profile(
    key="group",
    label="Group photo",
    blurb="Several people. The group as a whole is the subject, so centering is fine.",
    applies_when="Two or more people photographed together, posed or candid.",
    thirds_tol=None,  # the cluster belongs centered, not on a third
    headroom=(0.04, 0.16),
    center_is_good=True,
    size_range=(0.15, 0.85),
    weights={"edges": 1.6, "headroom": 1.2},  # do not crop anyone
)

SYMMETRY = Profile(
    key="symmetry",
    label="Symmetry / architecture",
    blurb="Centering is the point. Level and balance are critical; thirds does not apply.",
    applies_when=(
        "Architecture, interiors, reflections, corridors, facades, or any composition "
        "built on a central axis or mirrored halves."
    ),
    thirds_tol=None,  # explicitly inapplicable
    tilt_tol_deg=0.8,  # a symmetric frame shows tilt mercilessly
    headroom=None,
    pitch_tol=0.04,
    balance_tol=0.10,
    size_range=None,
    yaw_tol=None,
    center_is_good=True,
    symmetry_matters=True,
    weights={"tilt": 1.8, "balance": 2.0, "camera pitch": 1.5},
)

LANDSCAPE = Profile(
    key="landscape",
    label="Landscape",
    blurb="Horizon placement and level dominate. Subject size barely matters.",
    applies_when="A wide natural or urban scene where terrain, sky, or water is the subject.",
    thirds_tol=0.08,  # the horizon wants a third, loosely
    tilt_tol_deg=0.7,  # a tilted sea horizon is the classic error
    headroom=None,
    pitch_tol=0.10,
    size_range=None,
    yaw_tol=None,
    weights={"tilt": 2.0, "balance": 0.8, "edges": 0.4},
)

STREET = Profile(
    key="street",
    label="Street / documentary",
    blurb="The moment outranks the geometry. Tilt and tight crops are idiomatic.",
    applies_when=(
        "Candid public life, reportage, or documentary work where timing and content "
        "matter more than precise framing."
    ),
    thirds_tol=0.12,
    tilt_tol_deg=6.0,  # a cant is a deliberate street idiom
    headroom=(0.0, 0.30),
    pitch_tol=None,
    edge_tol=0.0,  # clipped edges are part of the language
    size_range=(0.02, 0.90),
    balance_tol=0.40,
    weights={"thirds": 0.5, "tilt": 0.3, "balance": 0.4, "edges": 0.2},
)

ACTION = Profile(
    key="action",
    label="Action / sports",
    blurb="Room in the direction of travel, and do not clip the limbs.",
    applies_when="Sport, motion, animals or vehicles in movement, decisive-moment action.",
    thirds_tol=0.09,
    tilt_tol_deg=5.0,
    headroom=(0.0, 0.25),
    pitch_tol=None,
    edge_tol=0.012,
    size_range=(0.05, 0.75),
    weights={"looking room": 2.0, "edges": 1.5, "tilt": 0.4},
)

MINIMAL = Profile(
    key="minimal",
    label="Minimalist / negative space",
    blurb="A small subject in a large emptiness is the goal, not an error.",
    applies_when=(
        "A deliberately sparse frame — one small subject against a large plain field "
        "of sky, wall, water, snow or fog."
    ),
    thirds_tol=0.10,
    tilt_tol_deg=1.2,
    headroom=None,
    size_range=(0.002, 0.20),  # small IS the intent
    balance_tol=0.55,  # lopsided emptiness is the point
    yaw_tol=None,
    weights={"subject size": 0.6, "balance": 0.3, "tilt": 1.2},
)

PRODUCT = Profile(
    key="product",
    label="Product / flat lay",
    blurb="Centered, square-on, level. Thirds does not apply.",
    applies_when="A product, object, dish, or flat-lay arrangement shot for clarity.",
    thirds_tol=None,
    tilt_tol_deg=0.8,
    headroom=None,
    pitch_tol=0.05,
    size_range=(0.15, 0.85),
    balance_tol=0.15,
    yaw_tol=None,
    center_is_good=True,
    symmetry_matters=True,
    weights={"tilt": 1.6, "balance": 1.4, "subject size": 1.2},
)

GENERIC = Profile(
    key="generic",
    label="No stated intent",
    blurb="v1's universal rule set. Kept as the baseline it is — it has no measured signal.",
    applies_when="Use only when the intent is genuinely unclear from the photo and description.",
)

PROFILES: dict[str, Profile] = {
    p.key: p
    for p in (PORTRAIT, GROUP, SYMMETRY, LANDSCAPE, STREET, ACTION, MINIMAL, PRODUCT, GENERIC)
}

# Keywords for the no-API-key fallback classifier. Deliberately plain: this is a
# backstop so the app works offline, not a pretence at language understanding.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "portrait": (
        "portrait", "headshot", "self portrait", "selfie", "face", "profile shot",
        "her face", "his face", "their face", "model", "graduation photo",
    ),
    "group": (
        "group", "family", "team", "friends", "everyone", "couple", "crowd of us",
        "wedding party", "class photo", "us together",
    ),
    "symmetry": (
        "symmetry", "symmetric", "symmetrical", "architecture", "architectural",
        "building", "facade", "reflection", "mirrored", "corridor", "hallway",
        "cathedral", "interior", "centered", "centred", "leading lines",
    ),
    "landscape": (
        "landscape", "horizon", "sunset", "sunrise", "mountain", "ocean", "sea",
        "beach", "field", "valley", "vista", "scenery", "skyline", "seascape",
    ),
    "street": (
        "street", "candid", "documentary", "reportage", "photojournalism",
        "everyday life", "passerby", "market", "city life", "unposed",
    ),
    "action": (
        "action", "sport", "sports", "running", "jumping", "motion", "movement",
        "match", "game", "race", "dancing", "skateboard", "surfing", "bird in flight",
    ),
    "minimal": (
        "minimal", "minimalist", "negative space", "empty", "emptiness", "sparse",
        "lonely", "isolated", "solitude", "vast", "simple composition", "fog", "snow",
    ),
    "product": (
        "product", "flat lay", "flatlay", "food", "dish", "meal", "object",
        "still life", "packshot", "for sale", "listing", "menu",
    ),
}


def catalog_for_prompt() -> str:
    """The profile menu, formatted for the LLM that picks one."""
    lines = []
    for p in PROFILES.values():
        disabled = [r for r in
                    ("thirds", "tilt", "headroom", "camera pitch", "edges",
                     "subject size", "balance", "looking room")
                    if not p.applies(r)]
        note = f" (ignores: {', '.join(disabled)})" if disabled else ""
        lines.append(f"- {p.key}: {p.applies_when}{note}")
    return "\n".join(lines)
