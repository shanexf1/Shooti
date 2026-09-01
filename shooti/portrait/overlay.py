"""Draw the portrait analysis so the photographer can check it themselves.

Two jobs. First, show the advice: eye line, thirds, predicted crop-safe zones.
Second — and the reason this matters — show the EVIDENCE. Face detectors produce
false positives (a toy playhouse in the test set reads as a face at 0.83
confidence), and predicted body landmarks rest on an upright-adult assumption
that a seated or lunging subject breaks. Drawing the landmarks lets a human
invalidate the analysis in one glance, which no confidence number does.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..subject import Face
from .anatomy import JOINT_ZONES, LANDMARKS, CropGeometry
from .pose import HeadPose

CYAN = (255, 214, 0)
MAGENTA = (200, 60, 255)
GREEN = (90, 220, 120)
RED = (70, 70, 245)
AMBER = (60, 190, 255)
WHITE = (255, 255, 255)


def _scaled(width: int) -> tuple[int, float]:
    return max(1, int(round(width / 700))), max(0.38, width / 1800)


def _label(img, text, org, color, scale, thick):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    x = max(2, min(x, img.shape[1] - tw - 6))
    y = max(th + 6, y)
    cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + base), (25, 25, 25), -1)
    cv2.putText(img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _head_ellipse_for_draw(face: Face):
    fx, fy, fw, fh = face.box
    return (int(fx + fw / 2.0), int(fy + fh * 0.42)), (int(fw * 0.80), int(fh * 0.95))


def _dashed(img, p0, p1, color, thick, dash=12):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    dist = float(np.hypot(*(p1 - p0)))
    if dist < 1:
        return
    steps = max(1, int(dist // dash))
    for i in range(0, steps, 2):
        a = p0 + (p1 - p0) * (i / steps)
        b = p0 + (p1 - p0) * (min(i + 1, steps) / steps)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thick, cv2.LINE_AA)


def render(
    bgr: np.ndarray,
    face: Face,
    pose: HeadPose,
    geo: CropGeometry,
    *,
    show_landmarks: bool = True,
    show_body: bool = True,
    show_eye_line: bool = True,
    show_pose: bool = True,
    background=None,  # BackgroundAnalysis, for v4.1
    show_background: bool = False,
) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    t, s = _scaled(w)

    if show_body:
        # Predicted body landmarks, so the upright-adult assumption is visible.
        for name, depth in LANDMARKS.items():
            if name in ("crown", "eyes"):
                continue
            y = geo.crown_y + depth * geo.head_height_px
            if not 0 <= y < h:
                continue
            in_joint = any(lo <= depth <= hi for lo, hi, _ in JOINT_ZONES)
            colour = RED if in_joint else GREEN
            _dashed(img, (0, int(y)), (w, int(y)), colour, t)
            _label(img, name, (6, int(y) - 4), colour, s * 0.8, t)

        # Where the frame actually cuts, and whether that is a joint.
        edge_colour = RED if geo.joint else GREEN
        cv2.line(img, (0, h - t * 2), (w, h - t * 2), edge_colour, t * 3, cv2.LINE_AA)
        cut = f"frame ends at {geo.heads_to_bottom:.2f} heads = {geo.crop_name}"
        if geo.joint:
            cut += f"  CUTS {geo.joint.upper()}"
        _label(img, cut, (6, h - t * 8), edge_colour, s, t)

    if show_eye_line:
        ex, ey = face.eye_level
        _dashed(img, (0, int(ey)), (w, int(ey)), MAGENTA, t + 1)
        third = h / 3.0
        _dashed(img, (0, int(third)), (w, int(third)), WHITE, t)
        _label(img, "upper third", (w - 130, int(third) - 4), WHITE, s * 0.8, t)
        _label(img, f"eye line ({ey / h * 100:.0f}% down)", (6, int(ey) - 4), MAGENTA, s * 0.85, t)
        cv2.circle(img, (int(ex), int(ey)), t * 3, MAGENTA, -1, cv2.LINE_AA)

    fx, fy, fw, fh = face.box
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), CYAN, t + 1, cv2.LINE_AA)

    if show_landmarks:
        # The evidence. If these five dots are not on a face, nothing else holds.
        for i, lm in enumerate(face.landmarks):
            cv2.circle(img, (int(lm[0]), int(lm[1])), max(2, t * 2), AMBER, -1, cv2.LINE_AA)
        _label(img, f"face {face.score:.2f}", (fx, max(14, fy - 5)), CYAN, s * 0.85, t)

    if show_background and background is not None:
        # The halo ring is where separation is judged, so show it: a complaint
        # about "tone behind the head" should point at the pixels it measured.
        centre, axes = _head_ellipse_for_draw(face)
        cv2.ellipse(img, centre, (int(axes[0] * 1.75), int(axes[1] * 1.75)), 0, 0, 360,
                    MAGENTA, t, cv2.LINE_AA)
        _label(img, f"halo L {background.halo_luma:.0f} vs face {background.face_luma:.0f}",
               (centre[0] - int(axes[0] * 1.75), centre[1] - int(axes[1] * 1.75) - 6),
               MAGENTA, s * 0.8, t)

        for spot in background.hotspots:
            r = max(6, int((spot.area_frac * img.shape[0] * img.shape[1]) ** 0.5 / 2))
            cv2.circle(img, (int(spot.x), int(spot.y)), r, RED, t + 1, cv2.LINE_AA)
            _label(img, f"blown {spot.area_frac * 100:.1f}%",
                   (int(spot.x) - 30, int(spot.y) - r - 4), RED, s * 0.75, t)

        for spot in background.saturated_patches:
            cv2.circle(img, (int(spot.x), int(spot.y)), 12, AMBER, t + 1, cv2.LINE_AA)

        if background.vertical_intrusion_x is not None:
            x = int(background.vertical_intrusion_x)
            cv2.line(img, (x, 0), (x, h), RED, t + 1, cv2.LINE_AA)
            _label(img, "vertical into head", (x + 4, 20), RED, s * 0.8, t)
        if background.horizontal_intrusion_y is not None:
            y = int(background.horizontal_intrusion_y)
            cv2.line(img, (0, y), (w, y), RED, t + 1, cv2.LINE_AA)
            _label(img, "horizontal across subject", (6, y - 6), RED, s * 0.8, t)

    if show_pose and pose.ok:
        ex, ey = face.eye_level
        length = max(20.0, face.eye_distance * 1.6)
        yaw_r = np.radians(pose.yaw_deg)
        pitch_r = np.radians(pose.pitch_deg)
        tip = (int(ex + length * np.sin(yaw_r)), int(ey + length * np.sin(pitch_r)))
        cv2.arrowedLine(img, (int(ex), int(ey)), tip, CYAN, t + 1, cv2.LINE_AA, tipLength=0.3)
        _label(
            img,
            f"yaw {pose.yaw_deg:+.0f}  pitch {pose.pitch_deg:+.0f}  roll {pose.roll_deg:+.0f}",
            (fx, fy + fh + int(18 * s * 2)),
            CYAN, s * 0.85, t,
        )
    elif show_pose:
        _label(img, "pose: not reliable", (fx, fy + fh + 16), AMBER, s * 0.85, t)

    return img


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
