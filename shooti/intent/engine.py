"""Run a rule set against measurements.

Operates on the 18-d feature vector rather than on an image. That is a deliberate
choice: the same code path grades a live photo and replays over the 25,547 cached
AVA feature vectors, so the claim "intent-conditioned rules beat universal rules"
can actually be measured instead of asserted. See intent/experiment.py.

Direction of travel (pan left vs right) needs signed values that the feature
vector does not carry for roll, so the signed horizon angle can be passed in when
a live photo is at hand. Without it the engine still scores, and simply omits the
direction from the advice text rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grader.features import FEATURE_NAMES
from ..rules import Finding
from .profiles import GENERIC, Profile

IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}
THIRDS_POINTS = ((1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3))


@dataclass
class Verdict:
    profile: Profile
    findings: list[Finding]
    score: int
    skipped: list[str]  # rules this intent declares inapplicable

    @property
    def problems(self) -> list[Finding]:
        return [f for f in self.findings if f.severity != "ok"]


def _sev(excess: float, major_at: float) -> str:
    return "major" if excess >= major_at else "minor"


def _thirds(f: np.ndarray, p: Profile) -> Finding:
    if f[IDX["has_face"]] == 0 and f[IDX["subject_conf"]] == 0:
        return Finding("thirds", "ok", "No subject located, so placement can't be judged.",
                       "Nothing to correct.")
    ax, ay = float(f[IDX["anchor_x"]]), float(f[IDX["anchor_y"]])

    if p.center_is_good:
        # Inverted: this intent wants the subject centered.
        off = float(np.hypot(ax - 0.5, ay - 0.5))
        tol = p.thirds_tol or 0.06
        if off <= tol:
            return Finding("thirds", "ok",
                           f"Subject sits {off * 100:.0f}% from center — centered, as this intent wants.",
                           "Hold it.", data={"off_center": off})
        dx, dy = 0.5 - ax, 0.5 - ay
        moves = []
        if abs(dx) > tol * 0.6:
            moves.append(f"pan {'left' if dx > 0 else 'right'} about {abs(dx) * 100:.0f}%")
        if abs(dy) > tol * 0.6:
            moves.append(f"{'raise' if dy > 0 else 'lower'} the camera about {abs(dy) * 100:.0f}%")
        return Finding("thirds", _sev(off - tol, 0.10),
                       f"Subject is {off * 100:.0f}% off center, which undercuts a centered composition.",
                       ("To center it: " + ", then ".join(moves) + ".") if moves else "Recenter.",
                       penalty=min(26.0, (off - tol) * 120.0), data={"off_center": off})

    tol = p.thirds_tol or 0.055
    tx, ty = min(THIRDS_POINTS, key=lambda q: (q[0] - ax) ** 2 + (q[1] - ay) ** 2)
    dx, dy = tx - ax, ty - ay
    off = float(np.hypot(dx, dy))
    if off <= tol:
        return Finding("thirds", "ok", "Subject sits near a rule-of-thirds intersection.",
                       "Hold this framing.", data={"offset": off})
    moves = []
    if abs(dx) > tol * 0.6:
        moves.append(f"pan {'left' if dx > 0 else 'right'} about {abs(dx) * 100:.0f}% of the frame width")
    if abs(dy) > tol * 0.6:
        # Raising the camera moves the subject DOWN in frame.
        moves.append(f"{'raise' if dy > 0 else 'lower'} the camera about {abs(dy) * 100:.0f}% of the frame height")
    return Finding("thirds", _sev(off - tol, 0.09),
                   f"Subject is {off * 100:.0f}% of the frame off the nearest thirds intersection.",
                   ("To fix: " + ", then ".join(moves) + ".") if moves else "Move onto a thirds line.",
                   penalty=min(26.0, (off - tol) * 120.0), data={"offset": off})


def _tilt(f: np.ndarray, p: Profile, signed: float | None) -> Finding:
    if f[IDX["horizon_present"]] == 0 or f[IDX["horizon_strength"]] < 0.25:
        return Finding("tilt", "ok", "No strong horizon or long straight edge to check level against.",
                       "Nothing to correct.")
    angle = float(f[IDX["horizon_angle_abs"]]) * 20.0  # feature is abs(deg)/20
    tol = p.tilt_tol_deg or 1.5
    if angle <= tol:
        return Finding("tilt", "ok", f"Level within {angle:.1f}°.", "Hold it level.",
                       data={"angle_abs": angle})
    if signed is None:
        fix = f"Straighten by about {angle:.1f}°."
    else:
        direction = "counter-clockwise" if signed > 0 else "clockwise"
        tip = "left" if signed > 0 else "right"
        fix = f"Roll the camera {angle:.1f}° {direction} (tip the top edge {tip})."
    return Finding("tilt", _sev(angle - tol, 3.0),
                   f"Horizon is off level by {angle:.1f}°, past the {tol:.1f}° this intent allows.",
                   fix, penalty=min(24.0, (angle - tol) * 3.5), data={"angle_abs": angle})


def _headroom(f: np.ndarray, p: Profile) -> Finding:
    if f[IDX["has_face"]] == 0:
        return Finding("headroom", "ok", "No face detected, so headroom doesn't apply.",
                       "Nothing to correct.")
    gap = float(f[IDX["headroom"]])
    low, high = p.headroom or (0.03, 0.14)
    if gap < 0:
        return Finding("headroom", "major", "The top of the head is cut off by the frame.",
                       "Tilt up or step back until the whole head is inside.", penalty=18.0)
    if low <= gap <= high:
        return Finding("headroom", "ok", f"Headroom {gap * 100:.0f}% — in range for this intent.",
                       "Hold this height.", data={"gap": gap})
    if gap > high:
        return Finding("headroom", _sev(gap - high, 0.12),
                       f"Too much headroom — {gap * 100:.0f}% of the frame is empty above the subject.",
                       f"Tilt down or lower the camera so about {int(high * 100)}% is left above the head.",
                       penalty=min(20.0, (gap - high) * 90.0), data={"gap": gap})
    return Finding("headroom", "minor",
                   f"Headroom is tight at {gap * 100:.0f}%; the head nearly touches the top edge.",
                   "Tilt up slightly or step back.", penalty=8.0, data={"gap": gap})


def _pitch(f: np.ndarray, p: Profile) -> Finding:
    if f[IDX["horizon_present"]] == 0 or f[IDX["horizon_strength"]] < 0.35:
        return Finding("camera pitch", "ok", "Not enough horizon signal to estimate pitch.",
                       "Nothing to correct.")
    ratio = 0.5 - float(f[IDX["horizon_y"]])  # >0 => horizon above center
    tol = p.pitch_tol or 0.06
    if abs(ratio) <= tol:
        return Finding("camera pitch", "ok", "Camera is close to level in pitch.", "Hold this angle.",
                       data={"ratio": ratio})
    down = ratio > 0
    return Finding("camera pitch", "minor",
                   f"Horizon sits {'above' if down else 'below'} center, so the camera is angled "
                   f"{'down' if down else 'up'}.",
                   ("Raise the lens or tilt up" if down else "Lower the lens or tilt down")
                   + " — or commit to the angle deliberately.",
                   penalty=min(13.0, (abs(ratio) - tol) * 60.0), data={"ratio": ratio})


def _edges(f: np.ndarray, p: Profile) -> Finding:
    frac = float(f[IDX["edge_touch_frac"]])
    n = int(round(frac * 4))
    if n == 0:
        return Finding("edges", "ok", "Subject is clear of the frame edges.", "Nothing to correct.")
    return Finding("edges", "major" if n > 1 else "minor",
                   f"Subject touches {n} frame edge{'s' if n > 1 else ''}.",
                   "Step back or pan away to keep the subject whole.",
                   penalty=7.0 * n, data={"edges": n})


def _size(f: np.ndarray, p: Profile) -> Finding:
    frac = float(f[IDX["subject_area_frac"]])
    low, high = p.size_range or (0.06, 0.62)
    if low <= frac <= high:
        return Finding("subject size", "ok",
                       f"Subject fills {frac * 100:.0f}% of the frame — right for this intent.",
                       "Nothing to correct.", data={"fraction": frac})
    if frac < low:
        return Finding("subject size", "minor",
                       f"Subject fills only {frac * 100:.0f}% and reads as small for this intent.",
                       "Move closer or zoom in.", penalty=min(14.0, (low - frac) * 180.0),
                       data={"fraction": frac})
    return Finding("subject size", "minor",
                   f"Subject fills {frac * 100:.0f}%, leaving little breathing room.",
                   "Step back or zoom out.", penalty=min(14.0, (frac - high) * 40.0),
                   data={"fraction": frac})


def _balance(f: np.ndarray, p: Profile) -> Finding:
    skew = float(f[IDX["balance_skew"]])
    tol = p.balance_tol or 0.22
    if abs(skew) <= tol:
        return Finding("balance", "ok", "Visual weight is balanced left to right.",
                       "Nothing to correct.", data={"skew": skew})
    heavy = "right" if skew > 0 else "left"
    empty = "left" if skew > 0 else "right"
    msg = (f"The halves are mismatched — detail piles up on the {heavy}."
           if p.symmetry_matters
           else f"Detail is piled up on the {heavy}; the {empty} side reads as empty.")
    return Finding("balance", _sev(abs(skew) - tol, 0.25), msg,
                   f"Pan toward the {heavy}, or move the subject toward the {empty} to counterweight.",
                   penalty=min(16.0, (abs(skew) - tol) * 50.0), data={"skew": skew})


def _looking_room(f: np.ndarray, p: Profile) -> Finding:
    if f[IDX["has_face"]] == 0:
        return Finding("looking room", "ok", "No face detected, so there's no gaze to balance.",
                       "Nothing to correct.")
    yaw = float(f[IDX["head_yaw_abs"]])
    tol = p.yaw_tol or 0.15
    if yaw < tol:
        return Finding("looking room", "ok", "Subject faces the camera; side space can stay even.",
                       "Nothing to correct.", data={"yaw_abs": yaw})
    ahead = float(f[IDX["space_ahead"]])
    if ahead >= 0.55:
        return Finding("looking room", "ok", "Subject is turned, and has the open space ahead of them.",
                       "Nothing to correct.", data={"space_ahead": ahead})
    return Finding("looking room", _sev(0.55 - ahead, 0.15),
                   f"Subject is turned but only {ahead * 100:.0f}% of the frame is open that way, "
                   "so their gaze runs into the edge.",
                   "Pan to open space in front of the face.",
                   penalty=min(18.0, (0.55 - ahead) * 75.0), data={"space_ahead": ahead})


RULES = {
    "thirds": _thirds,
    "tilt": _tilt,
    "headroom": _headroom,
    "camera pitch": _pitch,
    "edges": _edges,
    "subject size": _size,
    "balance": _balance,
    "looking room": _looking_room,
}


def evaluate(
    features: np.ndarray,
    profile: Profile = GENERIC,
    *,
    horizon_angle_signed: float | None = None,
) -> Verdict:
    findings: list[Finding] = []
    skipped: list[str] = []

    for rule, fn in RULES.items():
        if not profile.applies(rule):
            skipped.append(rule)
            continue
        finding = fn(features, profile, horizon_angle_signed) if rule == "tilt" else fn(features, profile)
        finding.penalty *= profile.weight(rule)
        findings.append(finding)

    score = int(round(max(0.0, 100.0 - sum(f.penalty for f in findings))))
    order = {"major": 0, "minor": 1, "ok": 2}
    findings.sort(key=lambda f: (order[f.severity], -f.penalty))
    return Verdict(profile, findings, score, skipped)


def score_only(features: np.ndarray, profile: Profile) -> float:
    """Just the number — the hot path for the dataset-wide experiment."""
    total = 0.0
    for rule, fn in RULES.items():
        if not profile.applies(rule):
            continue
        finding = fn(features, profile, None) if rule == "tilt" else fn(features, profile)
        total += finding.penalty * profile.weight(rule)
    return max(0.0, 100.0 - total)
