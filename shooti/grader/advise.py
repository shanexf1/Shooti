"""Advice by counterfactual search, not by rule violation.

This is the mechanism that fixes v1's core flaw. v1 asserted "eye level is not on
a thirds intersection, therefore this is worse". v2 asks a different question:

    if I actually reframed the shot this way, would humans rate it higher?

It builds a set of candidate reframings (shift, tighten, roll), grades each with
the learned model, and reports the ones that beat the current frame. A centered
or symmetric photo that cannot be improved by shifting simply produces no shift
suggestion — no rule has to be special-cased for it.

Honest limitation: we can only crop *inside* the frame that exists. "Step back"
or "zoom out" cannot be simulated from a single photo, since the pixels outside
the frame were never captured. So the search covers pans, tightening, and roll,
and the UI says so rather than implying full coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .features import geometric_features
from .grade import Grade, grade_batch

# Rotated candidates crop well inside the frame so no reflected or black pixels
# enter the model — fake content would produce fake scores.
ROLL_SCALE = 0.74
SHIFT_SCALE = 0.86
ZOOM_SCALES = (0.72, 0.60)
MIN_GAIN = 0.04  # below this, a "improvement" is model noise, not advice


@dataclass
class Candidate:
    label: str
    kind: str  # "keep" | "pan" | "tighten" | "roll"
    scale: float
    cx: float  # crop center, fraction of width
    cy: float
    angle: float  # degrees, positive = counter-clockwise image rotation
    advice: str


@dataclass
class Suggestion:
    candidate: Candidate
    grade: Grade
    gain: float
    crop_bgr: np.ndarray


def _crop(bgr: np.ndarray, cand: Candidate) -> np.ndarray:
    h, w = bgr.shape[:2]
    src = bgr
    if abs(cand.angle) > 1e-6:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), cand.angle, 1.0)
        src = cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR)

    cw, ch = cand.scale * w, cand.scale * h
    # Clamp the window inside the frame so no synthetic pixels are ever graded.
    x0 = min(max(cand.cx * w - cw / 2.0, 0.0), w - cw)
    y0 = min(max(cand.cy * h - ch / 2.0, 0.0), h - ch)
    return src[int(y0) : int(y0 + ch), int(x0) : int(x0 + cw)].copy()


def candidates() -> list[Candidate]:
    out = [
        Candidate("as shot", "keep", 1.0, 0.5, 0.5, 0.0, "Keep this framing."),
    ]

    # Pans: move the crop window, which is what panning the camera does. A window
    # moved right shows more of the right side, i.e. the camera panned right.
    step = (1.0 - SHIFT_SCALE) / 2.0
    for dx, dy, label, advice in (
        (+step, 0.0, "pan right", "Pan the camera right"),
        (-step, 0.0, "pan left", "Pan the camera left"),
        (0.0, +step, "tilt down", "Tilt or lower the camera"),
        (0.0, -step, "tilt up", "Tilt or raise the camera"),
        (+step, -step, "pan right + up", "Pan right and raise the camera"),
        (-step, -step, "pan left + up", "Pan left and raise the camera"),
        (+step, +step, "pan right + down", "Pan right and lower the camera"),
        (-step, +step, "pan left + down", "Pan left and lower the camera"),
    ):
        pct = int(round(abs(dx or dy) * 100))
        out.append(
            Candidate(
                label,
                "pan",
                SHIFT_SCALE,
                0.5 + dx,
                0.5 + dy,
                0.0,
                f"{advice} about {pct}% of the frame",
            )
        )

    # Tighter framings: what moving closer or zooming in would give you.
    for s in ZOOM_SCALES:
        pct = int(round((1.0 - s) * 100))
        out.append(
            Candidate(
                f"tighten {pct}%",
                "tighten",
                s,
                0.5,
                0.5,
                0.0,
                f"Move closer or zoom in about {pct}%",
            )
        )

    # Roll: straighten or deliberately cant the frame.
    for angle in (-6.0, -3.0, 3.0, 6.0):
        direction = "counter-clockwise" if angle > 0 else "clockwise"
        out.append(
            Candidate(
                f"roll {angle:+.0f}°",
                "roll",
                ROLL_SCALE,
                0.5,
                0.5,
                angle,
                f"Roll the camera {abs(angle):.0f}° {direction}",
            )
        )
    return out


def suggest(
    bgr: np.ndarray,
    channels: str = "both",
    top_k: int = 3,
    min_gain: float = MIN_GAIN,
) -> tuple[Grade, list[Suggestion]]:
    """Grade the frame, then return the reframings that actually score higher.

    Returns (grade_as_shot, suggestions). An empty suggestion list is a real
    result: the model could not find a reframing worth making.
    """
    cands = candidates()
    crops = [_crop(bgr, c) for c in cands]
    geo = np.stack([geometric_features(c) for c in crops])
    grades = grade_batch(crops, channels=channels, geo=geo)

    base = grades[0]  # index 0 is "as shot" by construction
    assert cands[0].kind == "keep"

    scored = [
        Suggestion(c, g, g.score - base.score, crop)
        for c, g, crop in zip(cands[1:], grades[1:], crops[1:])
    ]
    scored.sort(key=lambda s: s.gain, reverse=True)
    return base, [s for s in scored if s.gain >= min_gain][:top_k]


def response_range(bgr: np.ndarray, channels: str = "both") -> float:
    """Spread of predicted scores across all candidates.

    A diagnostic, not advice: if this is near zero the model is blind to framing
    changes and the suggestions above are meaningless. Worth checking before
    trusting any of this.
    """
    cands = candidates()
    crops = [_crop(bgr, c) for c in cands]
    geo = np.stack([geometric_features(c) for c in crops])
    scores = np.array([g.score for g in grade_batch(crops, channels=channels, geo=geo)])
    return float(scores.max() - scores.min())
