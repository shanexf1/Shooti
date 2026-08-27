"""Geometric measurements as a feature vector — no penalties, no thresholds.

This is the deliberate difference from v1. `shooti/rules.py` turns the same
measurements into penalties using hand-set tolerances, which is what made v1
insist that an off-thirds eye position is a defect. Here the numbers are handed
to a model as features and it learns what they're worth from human ratings.

The vector order is fixed and must not be reordered — checkpoints depend on it.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..rules import Analysis, Horizon, analyze, detect_horizon
from ..subject import Subject, detect_subject

FEATURE_NAMES: tuple[str, ...] = (
    "has_face",
    "face_count_log",
    "anchor_x",
    "anchor_y",
    "thirds_dist",
    "center_dist",
    "subject_area_frac",
    "subject_conf",
    "headroom",
    "head_yaw_abs",
    "space_ahead",
    "horizon_present",
    "horizon_angle_abs",
    "horizon_strength",
    "horizon_y",
    "balance_skew",
    "edge_touch_frac",
    "aspect_log",
)
N_FEATURES = len(FEATURE_NAMES)

THIRDS = ((1 / 3, 1 / 3), (2 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 2 / 3))


def _balance_skew(bgr: np.ndarray) -> float:
    w = bgr.shape[1]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    energy = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    left = float(energy[:, : w // 2].sum())
    right = float(energy[:, w // 2 :].sum())
    total = left + right
    return 0.0 if total <= 0 else (right - left) / total


def build_features(
    subject: Subject,
    horizon: Horizon | None,
    anchor: tuple[float, float],
    w: int,
    h: int,
    balance_skew: float,
) -> np.ndarray:
    """Assemble the vector from an already-computed detection pass.

    Detection is the expensive step, so training extracts the v1 rule score and
    these features from one pass rather than detecting twice.
    """
    ax, ay = anchor
    nx, ny = ax / w, ay / h

    thirds_dist = min(float(np.hypot(nx - tx, ny - ty)) for tx, ty in THIRDS)
    center_dist = float(np.hypot(nx - 0.5, ny - 0.5))

    face = subject.face
    if face is not None:
        headroom = float(subject.top / h)
        yaw = float(face.yaw)
        space_ahead = (1.0 - nx) if yaw > 0 else nx
    else:
        headroom = 0.0
        yaw = 0.0
        space_ahead = 0.5

    x, y, bw, bh = subject.box
    edges = sum(
        (
            x <= 1,
            (x + bw) >= w - 1,
            y <= 1,
            (y + bh) >= h - 1,
        )
    )

    vec = np.array(
        [
            1.0 if face is not None else 0.0,
            float(np.log1p(subject.face_count)),
            nx,
            ny,
            thirds_dist,
            center_dist,
            float(subject.area_fraction),
            float(subject.confidence),
            headroom,
            abs(yaw),
            float(space_ahead),
            1.0 if horizon is not None else 0.0,
            (abs(horizon.angle_deg) / 20.0) if horizon else 0.0,
            float(horizon.strength) if horizon else 0.0,
            (horizon.y / h) if horizon else 0.5,
            float(balance_skew),
            edges / 4.0,
            float(np.log(w / max(h, 1))),
        ],
        dtype=np.float32,
    )
    assert vec.shape == (N_FEATURES,), vec.shape
    return vec


def features_from_analysis(analysis: Analysis, bgr: np.ndarray | None = None) -> np.ndarray:
    """Features from a v1 Analysis, reusing its detection pass."""
    skew = 0.0
    for f in analysis.findings:
        if f.rule == "balance":
            skew = float(f.data.get("skew", 0.0))
            break
    return build_features(
        analysis.subject,
        analysis.horizon,
        analysis.anchor,
        analysis.width,
        analysis.height,
        skew,
    )


def geometric_features(bgr: np.ndarray, download_model: bool = True) -> np.ndarray:
    """Measure the frame from scratch. Returns float32 of length N_FEATURES."""
    h, w = bgr.shape[:2]
    subject = detect_subject(bgr, download_model=download_model)
    horizon = detect_horizon(bgr)
    return build_features(subject, horizon, subject.anchor, w, h, _balance_skew(bgr))
