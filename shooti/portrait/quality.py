"""Portrait-specific image quality: eye focus, face light, background behind the head.

These are rules only a portrait tool can have. "Is the near eye sharp" is
meaningless for a landscape and non-negotiable for a portrait; "is something
growing out of their head" only exists as a concept because of portraiture.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..subject import Face


@dataclass
class EyeFocus:
    eye_sharpness: float  # normalized Laplacian energy at the eyes
    face_sharpness: float
    background_sharpness: float
    eye_is_sharpest: bool
    ratio_to_background: float


def _sharpness(patch: np.ndarray) -> float:
    if patch.size < 16:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    # Normalize by local contrast so a dark patch is not called soft.
    denom = float(gray.std()) + 1e-3
    return float(lap.var() ** 0.5 / denom)


def _patch(bgr: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray:
    h, w = bgr.shape[:2]
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r))
    return bgr[y0:y1, x0:x1]


def eye_focus(bgr: np.ndarray, face: Face) -> EyeFocus:
    """Is the eye region the sharpest part of the frame?

    The portrait rule is that the near eye must be sharp; a soft eye with a sharp
    ear or sharp background is the classic focus miss.
    """
    r = max(4.0, face.eye_distance * 0.35)
    eyes = [_patch(bgr, *e, r) for e in (face.eye_a, face.eye_b)]
    eye_s = max((_sharpness(p) for p in eyes), default=0.0)

    fx, fy, fw, fh = face.box
    face_s = _sharpness(bgr[max(0, fy) : fy + fh, max(0, fx) : fx + fw])

    # Background: the frame with the face region blanked out.
    mask = np.ones(bgr.shape[:2], np.uint8)
    pad_x, pad_y = int(0.8 * fw), int(0.8 * fh)
    mask[max(0, fy - pad_y) : fy + fh + pad_y, max(0, fx - pad_x) : fx + fw + pad_x] = 0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if mask.sum() > 64:
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        vals = lap[mask == 1]
        bg_s = float(vals.std() / (gray[mask == 1].std() + 1e-3))
    else:
        bg_s = 0.0

    return EyeFocus(
        eye_sharpness=eye_s,
        face_sharpness=face_s,
        background_sharpness=bg_s,
        eye_is_sharpest=eye_s >= bg_s,
        ratio_to_background=eye_s / (bg_s + 1e-6),
    )


@dataclass
class FaceLight:
    mean: float  # 0-255 on the face
    side_ratio: float  # >0 means the frame-right side of the face is brighter
    top_ratio: float  # >0 means the forehead is brighter than the chin
    clipped_highlight: float  # fraction of face pixels blown out
    crushed_shadow: float  # fraction of face pixels at black
    background_mean: float

    @property
    def direction(self) -> str:
        if abs(self.side_ratio) < 0.06:
            return "frontal / flat"
        return "from frame right" if self.side_ratio > 0 else "from frame left"


def face_light(bgr: np.ndarray, face: Face) -> FaceLight:
    fx, fy, fw, fh = face.box
    h, w = bgr.shape[:2]
    x0, x1 = max(0, fx), min(w, fx + fw)
    y0, y1 = max(0, fy), min(h, fy + fh)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return FaceLight(0, 0, 0, 0, 0, 0)

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0].astype(np.float32) * (255.0 / 100.0) if lab[:, :, 0].max() <= 100 else lab[:, :, 0].astype(np.float32)
    mean = float(lum.mean())

    half = lum.shape[1] // 2
    left, right = float(lum[:, :half].mean()), float(lum[:, half:].mean())
    side = (right - left) / max(right + left, 1e-6)

    vhalf = lum.shape[0] // 2
    top, bottom = float(lum[:vhalf].mean()), float(lum[vhalf:].mean())
    top_ratio = (top - bottom) / max(top + bottom, 1e-6)

    clipped = float((lum >= 250).mean())
    crushed = float((lum <= 8).mean())

    mask = np.ones(bgr.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = 0
    full_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    bg_mean = float(full_lab[mask == 1].mean()) if mask.sum() > 16 else mean

    return FaceLight(mean, side, top_ratio, clipped, crushed, bg_mean)


@dataclass
class HeadBackground:
    clutter: float  # edge density in a band around the head, 0-1
    intrusion: bool  # a strong vertical line crossing just above the head
    intrusion_x: float | None


def head_background(bgr: np.ndarray, face: Face) -> HeadBackground:
    """Looks for busy background behind the head, and the classic pole-through-the-head."""
    h, w = bgr.shape[:2]
    fx, fy, fw, fh = face.box
    band_y0 = int(max(0, fy - 1.2 * fh))
    band_y1 = int(max(1, fy + 0.15 * fh))
    band_x0 = int(max(0, fx - 0.5 * fw))
    band_x1 = int(min(w, fx + fw + 0.5 * fw))
    band = bgr[band_y0:band_y1, band_x0:band_x1]
    if band.size == 0:
        return HeadBackground(0.0, False, None)

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 170)
    clutter = float((edges > 0).mean())

    # A near-vertical line spanning most of the band, above the head, is a pole.
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=40,
        minLineLength=int(0.6 * band.shape[0]), maxLineGap=6,
    )
    intrusion, ix = False, None
    if lines is not None and len(lines):
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            if abs(x2 - x1) <= max(3, 0.06 * band.shape[1]):
                cand = band_x0 + (x1 + x2) / 2.0
                # Only counts if it sits behind the head, not off to the side.
                if fx - 0.15 * fw <= cand <= fx + fw + 0.15 * fw:
                    intrusion, ix = True, float(cand)
                    break
    return HeadBackground(clutter, intrusion, ix)
