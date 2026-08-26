"""Measure composition, then turn the measurements into actionable moves.

Every rule returns a Finding whose `action` is phrased as a camera move, plus a
penalty feeding a single 0-100 score. Nothing here calls an LLM — this layer has
to stand on its own.

Direction convention, stated once: panning the camera LEFT moves the frame left,
which makes the subject appear further RIGHT in it. So a subject sitting left of
its target needs a LEFT pan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .subject import Point, Subject, detect_subject

# Tolerances, as fractions of frame width/height unless noted.
THIRDS_TOL = 0.055
TILT_TOL_DEG = 1.5
HEADROOM_IDEAL = (0.03, 0.14)
PITCH_TOL = 0.06
EDGE_TOL = 0.008
SUBJECT_SIZE_RANGE = (0.06, 0.62)
YAW_TOL = 0.15


@dataclass
class Finding:
    rule: str
    severity: str  # "ok" | "minor" | "major"
    message: str
    action: str
    penalty: float = 0.0
    data: dict = field(default_factory=dict)


@dataclass
class Horizon:
    angle_deg: float  # positive = horizon rises toward the right of frame
    y: float  # pixel row where the line crosses the frame's center column
    strength: float  # 0-1 confidence that a real horizon exists


@dataclass
class Analysis:
    width: int
    height: int
    subject: Subject
    horizon: Horizon | None
    anchor: Point  # the point being composed (eye level, or subject center)
    target: Point  # where that point should sit
    findings: list[Finding]
    score: int

    @property
    def problems(self) -> list[Finding]:
        return [f for f in self.findings if f.severity != "ok"]


def detect_horizon(bgr: np.ndarray) -> Horizon | None:
    """Find the dominant near-horizontal line.

    Doubles as a camera-pitch estimate: for a level camera the true horizon
    projects onto the image's center row, so a horizon above center means the
    camera is pitched down, below center means pitched up.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=80,
        minLineLength=int(0.30 * w),
        maxLineGap=int(0.02 * w),
    )
    if lines is None or len(lines) == 0:
        return None
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    segments = np.asarray(lines).reshape(-1, 4)

    angles: list[float] = []
    centers: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in segments:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 0.30 * w or abs(dx) < 1e-6:
            continue
        angle = float(np.degrees(np.arctan2(-dy, dx)))  # image y grows downward
        if abs(angle) > 20.0:
            continue
        y_at_center = y1 + (dy / dx) * (w / 2.0 - x1)
        if not 0 <= y_at_center <= h:
            continue
        angles.append(angle)
        centers.append(float(y_at_center))
        weights.append(length)

    if not angles:
        return None

    weights_arr = np.asarray(weights)
    order = np.argsort(weights_arr)[::-1][:12]
    return Horizon(
        angle_deg=float(np.median(np.asarray(angles)[order])),
        y=float(np.median(np.asarray(centers)[order])),
        # One clean full-width line is already good evidence, so normalize so
        # that case lands around 0.5 rather than at the detection threshold.
        strength=float(min(1.0, weights_arr[order].sum() / (2.0 * w))),
    )


def _thirds_target(subject: Subject, anchor: Point, width: int, height: int) -> Point:
    """Nearest useful rule-of-thirds intersection.

    For people the eyes belong on the upper third line and the body on a side
    third, so y snaps to the upper third rather than the nearer one.
    """
    ax, ay = anchor
    tx = min((width / 3.0, 2.0 * width / 3.0), key=lambda v: abs(v - ax))
    if subject.face is not None:
        ty = height / 3.0
    else:
        ty = min((height / 3.0, 2.0 * height / 3.0), key=lambda v: abs(v - ay))
    return tx, ty


def _check_thirds(anchor: Point, target: Point, width: int, height: int, subject: Subject) -> Finding:
    if not subject.has_subject:
        return Finding(
            "thirds",
            "minor",
            "No dominant subject found — either the frame is very flat, or the subject "
            "doesn't stand out from its background.",
            "Pick one thing to be the subject and make it stand out, then place it on a thirds line.",
            penalty=12.0,
        )
    ax, ay = anchor
    tx, ty = target
    dx = (tx - ax) / width  # positive => subject must move right in frame
    dy = (ty - ay) / height  # positive => subject must move down in frame
    off = float(np.hypot(dx, dy))
    what = "Eye level" if subject.face else "Subject"

    if off <= THIRDS_TOL:
        return Finding(
            "thirds",
            "ok",
            f"{what} sits close to a rule-of-thirds intersection.",
            "Hold this framing.",
            data={"dx": dx, "dy": dy, "offset": off},
        )

    moves = []
    if abs(dx) > THIRDS_TOL * 0.7:
        moves.append(
            f"pan {'left' if dx > 0 else 'right'} about {abs(dx) * 100:.0f}% of the frame width"
        )
    if abs(dy) > THIRDS_TOL * 0.7:
        # dy > 0 means the subject must move DOWN in frame, which means the frame
        # moves up — so the camera goes up. Same inversion as the pan direction.
        moves.append(
            f"{'raise' if dy > 0 else 'lower'} the camera about {abs(dy) * 100:.0f}% of the frame height"
        )

    dead_center = abs(ax / width - 0.5) < 0.06 and abs(ay / height - 0.5) < 0.06
    message = (
        f"{what} is dead center, which flattens the composition."
        if dead_center
        else f"{what} is {off * 100:.0f}% of the frame off the nearest thirds intersection."
    )
    return Finding(
        "thirds",
        "major" if off > THIRDS_TOL * 2.2 else "minor",
        message,
        ("To fix: " + ", then ".join(moves) + ".") if moves else "Recenter on a thirds line.",
        penalty=min(28.0, off * 130.0),
        data={"dx": dx, "dy": dy, "offset": off},
    )


def _check_tilt(horizon: Horizon | None) -> Finding:
    if horizon is None or horizon.strength < 0.25:
        return Finding(
            "tilt",
            "ok",
            "No strong horizon or long straight edge to check level against.",
            "Nothing to correct.",
            data={"angle": None},
        )
    angle = horizon.angle_deg
    if abs(angle) <= TILT_TOL_DEG:
        return Finding(
            "tilt", "ok", f"Horizon is level within {abs(angle):.1f}°.", "Hold it level.",
            data={"angle": angle},
        )
    # A horizon rising to the right means the camera was rolled clockwise, so the
    # fix is a counter-clockwise roll — tip the camera's top edge left.
    high_side = "right" if angle > 0 else "left"
    direction = "counter-clockwise" if angle > 0 else "clockwise"
    tip = "left" if angle > 0 else "right"
    return Finding(
        "tilt",
        "major" if abs(angle) > 4.0 else "minor",
        f"Horizon is off level by {abs(angle):.1f}° — the {high_side} side sits higher.",
        f"Roll the camera {abs(angle):.1f}° {direction} (tip the top edge {tip}) to level it.",
        penalty=min(22.0, abs(angle) * 3.0),
        data={"angle": angle},
    )


def _check_headroom(subject: Subject, height: int) -> Finding:
    if subject.face is None:
        return Finding(
            "headroom", "ok", "Headroom only applies to people; no face detected.",
            "Nothing to correct.",
        )
    gap = subject.top / height
    low, high = HEADROOM_IDEAL

    if gap < 0:
        return Finding(
            "headroom",
            "major",
            "The top of the head is cut off by the frame.",
            "Tilt up or step back until the whole head is inside the frame.",
            penalty=18.0,
            data={"gap": gap},
        )
    if low <= gap <= high:
        return Finding(
            "headroom", "ok", f"Headroom is {gap * 100:.0f}% of frame height — in range.",
            "Hold this height.", data={"gap": gap},
        )
    if gap > high:
        return Finding(
            "headroom",
            "major" if gap > 0.25 else "minor",
            f"Too much headroom — {gap * 100:.0f}% of the frame is empty above the subject.",
            f"Tilt down or lower the camera so about {int(high * 100)}% is left above the head.",
            penalty=min(20.0, (gap - high) * 90.0),
            data={"gap": gap},
        )
    return Finding(
        "headroom",
        "minor",
        f"Headroom is tight at {gap * 100:.0f}%; the head nearly touches the top edge.",
        "Tilt up slightly or step back to give the head some air.",
        penalty=8.0,
        data={"gap": gap},
    )


def _check_looking_room(subject: Subject, width: int) -> Finding:
    """A turned head needs more space in the direction it faces."""
    if subject.face is None:
        return Finding(
            "looking room", "ok", "No face detected, so there's no gaze direction to balance.",
            "Nothing to correct.",
        )
    yaw = subject.face.yaw
    if abs(yaw) < YAW_TOL:
        return Finding(
            "looking room", "ok", "Subject faces the camera, so side space can stay even.",
            "Nothing to correct.", data={"yaw": yaw},
        )

    facing = "right" if yaw > 0 else "left"
    ax, _ = subject.anchor
    space_ahead = (1.0 - ax / width) if yaw > 0 else (ax / width)

    if space_ahead >= 0.55:
        return Finding(
            "looking room", "ok",
            f"Subject looks {facing} and has the open space on that side.",
            "Nothing to correct.", data={"yaw": yaw, "space_ahead": space_ahead},
        )
    return Finding(
        "looking room",
        "major" if space_ahead < 0.4 else "minor",
        f"Subject is turned {facing} but only {space_ahead * 100:.0f}% of the frame is open that way, "
        "so their gaze runs into the edge.",
        f"Pan {'right' if yaw > 0 else 'left'} to open up space in front of their face.",
        penalty=min(16.0, (0.55 - space_ahead) * 70.0),
        data={"yaw": yaw, "space_ahead": space_ahead},
    )


def _check_pitch(horizon: Horizon | None, height: int) -> Finding:
    if horizon is None or horizon.strength < 0.35:
        return Finding(
            "camera pitch", "ok", "Not enough horizon signal to estimate camera pitch.",
            "Nothing to correct.",
        )
    ratio = (height / 2.0 - horizon.y) / height  # >0 => horizon above center
    if abs(ratio) <= PITCH_TOL:
        return Finding(
            "camera pitch", "ok", "Camera looks close to level in pitch.", "Hold this angle.",
            data={"ratio": ratio},
        )
    penalty = min(12.0, (abs(ratio) - PITCH_TOL) * 60.0)
    if ratio > 0:
        return Finding(
            "camera pitch", "minor",
            "Horizon sits above center, so the camera is angled downward.",
            "Raise the lens or tilt up to bring the horizon toward the middle — "
            "or commit to the high angle deliberately.",
            penalty=penalty, data={"ratio": ratio},
        )
    return Finding(
        "camera pitch", "minor",
        "Horizon sits below center, so the camera is angled upward.",
        "Lower the lens or tilt down — or lean into the low angle for a heroic look.",
        penalty=penalty, data={"ratio": ratio},
    )


def _check_edges(subject: Subject, width: int, height: int) -> Finding:
    if not subject.has_subject:
        return Finding("edges", "ok", "No subject located, so cropping can't be judged.", "Nothing to correct.")
    x, y, w, h = subject.box
    touching = []
    if x / width <= EDGE_TOL:
        touching.append("left")
    if (x + w) / width >= 1 - EDGE_TOL:
        touching.append("right")
    if y / height <= EDGE_TOL:
        touching.append("top")
    if (y + h) / height >= 1 - EDGE_TOL:
        touching.append("bottom")

    if not touching:
        return Finding("edges", "ok", "Subject is clear of the frame edges.", "Nothing to correct.")
    return Finding(
        "edges",
        "major" if len(touching) > 1 else "minor",
        f"Subject is cropped hard at the {' and '.join(touching)} edge.",
        f"Step back or pan away from the {touching[0]} edge to keep the subject whole.",
        penalty=7.0 * len(touching),
        data={"edges": touching},
    )


def _check_subject_size(subject: Subject) -> Finding:
    if not subject.has_subject:
        return Finding("subject size", "ok", "No subject located, so size can't be judged.", "Nothing to correct.")
    frac = subject.area_fraction
    low, high = SUBJECT_SIZE_RANGE
    if low <= frac <= high:
        return Finding(
            "subject size", "ok", f"Subject fills {frac * 100:.0f}% of the frame — a readable size.",
            "Nothing to correct.", data={"fraction": frac},
        )
    if frac < low:
        return Finding(
            "subject size", "minor",
            f"Subject fills only {frac * 100:.0f}% of the frame and reads as small.",
            "Move closer or zoom in until it fills a quarter to a third of the frame.",
            penalty=min(14.0, (low - frac) * 180.0), data={"fraction": frac},
        )
    return Finding(
        "subject size", "minor",
        f"Subject fills {frac * 100:.0f}% of the frame, leaving little breathing room.",
        "Step back or zoom out to give the subject some negative space.",
        penalty=min(14.0, (frac - high) * 40.0), data={"fraction": frac},
    )


def _check_balance(bgr: np.ndarray) -> Finding:
    """Compare visual weight across the vertical midline using edge energy."""
    w = bgr.shape[1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    energy = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    left = float(energy[:, : w // 2].sum())
    right = float(energy[:, w // 2 :].sum())
    total = left + right
    if total <= 0:
        return Finding("balance", "ok", "Frame is too flat to judge balance.", "Nothing to correct.")

    skew = (right - left) / total  # >0 => right side is busier
    if abs(skew) <= 0.22:
        return Finding(
            "balance", "ok", "Visual weight is reasonably balanced left to right.",
            "Nothing to correct.", data={"skew": skew},
        )
    heavy = "right" if skew > 0 else "left"
    empty = "left" if skew > 0 else "right"
    return Finding(
        "balance", "minor",
        f"Detail is piled up on the {heavy}; the {empty} side reads as empty.",
        f"Pan toward the {heavy} to trim dead space, or place the subject on the {empty} "
        "so the halves counterweight.",
        penalty=min(12.0, (abs(skew) - 0.22) * 45.0),
        data={"skew": skew},
    )


def analyze(bgr: np.ndarray, download_model: bool = True) -> Analysis:
    h, w = bgr.shape[:2]
    subject = detect_subject(bgr, download_model=download_model)
    horizon = detect_horizon(bgr)
    anchor = subject.anchor
    # With no subject there is nothing to move, so target == anchor suppresses
    # the overlay's move arrow instead of pointing at an arbitrary third.
    target = _thirds_target(subject, anchor, w, h) if subject.has_subject else anchor

    findings = [
        _check_thirds(anchor, target, w, h, subject),
        _check_tilt(horizon),
        _check_headroom(subject, h),
        _check_looking_room(subject, w),
        _check_pitch(horizon, h),
        _check_edges(subject, w, h),
        _check_subject_size(subject),
        _check_balance(bgr),
    ]

    score = int(round(max(0.0, 100.0 - sum(f.penalty for f in findings))))
    order = {"major": 0, "minor": 1, "ok": 2}
    findings.sort(key=lambda f: (order[f.severity], -f.penalty))
    return Analysis(w, h, subject, horizon, anchor, target, findings, score)
