"""Draw the analysis back onto the photo so the advice is visible, not just read.

Layers, all optional so the user can toggle them:
  grid     - rule-of-thirds lines
  subject  - detected subject box (and face box when there is one)
  target   - crosshair where the subject center should sit
  arrow    - the move, drawn from where the subject is to where it should be
  horizon  - detected horizon plus a level reference
"""

from __future__ import annotations

import cv2
import numpy as np

from .rules import Analysis

WHITE = (255, 255, 255)
CYAN = (255, 214, 0)  # BGR
MAGENTA = (200, 60, 255)
GREEN = (90, 220, 120)
RED = (70, 70, 245)


def _scaled(width: int) -> tuple[int, float]:
    """Line thickness and font scale that hold up across image sizes."""
    thickness = max(1, int(round(width / 640)))
    return thickness, max(0.4, width / 1600)


def _dashed_line(img, p0, p1, color, thickness, dash=14):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    dist = float(np.hypot(*(p1 - p0)))
    if dist == 0:
        return
    steps = max(1, int(dist // dash))
    for i in range(steps):
        if i % 2:
            continue
        a = p0 + (p1 - p0) * (i / steps)
        b = p0 + (p1 - p0) * (min(i + 1, steps) / steps)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, thickness, cv2.LINE_AA)


def _label(img, text, org, color, scale, thickness):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + base), (25, 25, 25), -1)
    cv2.putText(
        img, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def render(
    bgr: np.ndarray,
    analysis: Analysis,
    *,
    grid: bool = True,
    subject: bool = True,
    target: bool = True,
    arrow: bool = True,
    horizon: bool = True,
) -> np.ndarray:
    img = bgr.copy()
    h, w = img.shape[:2]
    t, scale = _scaled(w)

    if grid:
        for i in (1, 2):
            x = int(w * i / 3)
            y = int(h * i / 3)
            _dashed_line(img, (x, 0), (x, h), WHITE, t)
            _dashed_line(img, (0, y), (w, y), WHITE, t)
        for i in (1, 2):
            for j in (1, 2):
                cv2.circle(img, (int(w * i / 3), int(h * j / 3)), t * 3, WHITE, -1, cv2.LINE_AA)

    if horizon and analysis.horizon and analysis.horizon.strength >= 0.25:
        hz = analysis.horizon
        slope = -np.tan(np.radians(hz.angle_deg))
        y0 = hz.y + slope * (0 - w / 2.0)
        y1 = hz.y + slope * (w - w / 2.0)
        off_level = abs(hz.angle_deg) > 1.5
        cv2.line(
            img, (0, int(y0)), (w, int(y1)), RED if off_level else GREEN, t + 1, cv2.LINE_AA
        )
        if off_level:
            _dashed_line(img, (0, int(hz.y)), (w, int(hz.y)), GREEN, t)
            _label(img, f"{hz.angle_deg:+.1f} deg", (12, int(hz.y) - 10), RED, scale, t)

    if subject and analysis.subject.has_subject:
        x, y, bw, bh = analysis.subject.box
        cv2.rectangle(img, (x, y), (x + bw, y + bh), CYAN, t + 1, cv2.LINE_AA)
        label = analysis.subject.kind
        if analysis.subject.face_count > 1:
            label = f"{analysis.subject.face_count} faces"
        _label(img, label, (x, max(20, y - 8)), CYAN, scale, t)

        face = analysis.subject.face
        if face is not None:
            fx, fy, fw, fh = face.box
            cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), CYAN, t, cv2.LINE_AA)
            for eye in (face.eye_a, face.eye_b):
                cv2.circle(img, (int(eye[0]), int(eye[1])), t * 2, CYAN, -1, cv2.LINE_AA)
            cv2.line(
                img,
                (int(face.eye_a[0]), int(face.eye_a[1])),
                (int(face.eye_b[0]), int(face.eye_b[1])),
                CYAN, t, cv2.LINE_AA,
            )

    cx, cy = analysis.anchor
    tx, ty = analysis.target

    if target and analysis.subject.has_subject:
        r = t * 8
        cv2.circle(img, (int(tx), int(ty)), r, MAGENTA, t + 1, cv2.LINE_AA)
        cv2.line(img, (int(tx - r * 1.6), int(ty)), (int(tx + r * 1.6), int(ty)), MAGENTA, t, cv2.LINE_AA)
        cv2.line(img, (int(tx), int(ty - r * 1.6)), (int(tx), int(ty + r * 1.6)), MAGENTA, t, cv2.LINE_AA)

    if arrow:
        dist = float(np.hypot(tx - cx, ty - cy))
        if dist > 0.04 * max(w, h):
            cv2.arrowedLine(
                img,
                (int(cx), int(cy)),
                (int(tx), int(ty)),
                MAGENTA,
                t + 2,
                cv2.LINE_AA,
                tipLength=min(0.35, 24.0 * t / dist),
            )

    badge = f"score {analysis.score}"
    color = GREEN if analysis.score >= 80 else (CYAN if analysis.score >= 60 else RED)
    _label(img, badge, (12, 30), color, scale * 1.3, t + 1)
    return img


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
