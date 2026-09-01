"""Background analysis for portraits.

Half a portrait is what is behind the subject, and it fails in specific,
recognisable ways. Each measure below corresponds to a mistake a portrait
photographer is taught to look for:

  separation   the subject's tone or colour matches what is right behind their
               head, so the silhouette dissolves into it
  depth        the background is as sharp as the face, so it competes
  hotspots     a blown-out patch (sky gap, window, lamp) pulls the eye off the face
  rivals       another face in the background, which the viewer will look at
  lines        a vertical growing out of the head, or a horizontal cutting the neck
  colour       a strongly saturated patch that outranks the skin
  escape       bright corners that lead the eye out of the frame
  clutter      general busy-ness competing with the face

Everything is measured in a ring immediately around the head, not across the
whole frame, because that is where the eye actually compares. A dark subject
against a dark wall is fine if the wall behind their head is light.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..subject import Face


@dataclass
class Hotspot:
    x: float
    y: float
    area_frac: float  # fraction of the frame
    distance_heads: float  # from the face centre, in head-heights
    brightness: float


@dataclass
class BackgroundAnalysis:
    face_luma: float
    halo_luma: float  # tone immediately around the head
    separation_luma: float  # 0-1, tonal contrast against the halo
    separation_color: float  # 0-1, LAB a/b distance against the halo
    blur_ratio: float  # face sharpness / background sharpness
    hotspots: list[Hotspot] = field(default_factory=list)
    saturated_patches: list[Hotspot] = field(default_factory=list)
    competing_faces: int = 0
    vertical_intrusion_x: float | None = None
    horizontal_intrusion_y: float | None = None
    corner_luma: float = 0.0
    clutter: float = 0.0
    head_mask: np.ndarray | None = None
    halo_mask: np.ndarray | None = None

    @property
    def merges(self) -> bool:
        """Subject silhouette dissolving into the background."""
        return self.separation_luma < 0.10 and self.separation_color < 0.10


def head_ellipse(face: Face, shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """Approximate the head silhouette, including hair above the detector box."""
    fx, fy, fw, fh = face.box
    cx = fx + fw / 2.0
    cy = fy + fh * 0.42  # the box starts near the brow, so the head centre is higher
    ax = fw * 0.80
    ay = fh * 0.95
    return (int(cx), int(cy)), (int(ax), int(ay))


def _masks(face: Face, shape: tuple[int, int], ring: float = 1.75):
    h, w = shape[:2]
    centre, axes = head_ellipse(face, shape)
    head = np.zeros((h, w), np.uint8)
    cv2.ellipse(head, centre, axes, 0, 0, 360, 255, -1)
    outer = np.zeros((h, w), np.uint8)
    cv2.ellipse(outer, centre, (int(axes[0] * ring), int(axes[1] * ring)), 0, 0, 360, 255, -1)
    halo = cv2.subtract(outer, head)
    background = cv2.bitwise_not(cv2.dilate(head, np.ones((9, 9), np.uint8)))
    return head, halo, background


def _sharpness_masked(gray: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() < 64 * 255:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sel = mask > 0
    contrast = float(gray[sel].std()) + 1e-3
    return float(lap[sel].std() / contrast)


def analyze(bgr: np.ndarray, face: Face, all_faces: list[Face] | None = None) -> BackgroundAnalysis:
    h, w = bgr.shape[:2]
    head, halo, background = _masks(face, bgr.shape)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    if L.max() <= 100:  # some builds scale L to 0-100
        L = L * 2.55
    A = lab[:, :, 1].astype(np.float32) - 128.0
    B = lab[:, :, 2].astype(np.float32) - 128.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    fx, fy, fw, fh = face.box
    face_sel = np.zeros((h, w), np.uint8)
    cv2.rectangle(face_sel, (max(0, fx), max(0, fy)), (min(w, fx + fw), min(h, fy + fh)), 255, -1)

    def mean_in(mask, arr):
        sel = mask > 0
        return float(arr[sel].mean()) if sel.any() else 0.0

    face_luma = mean_in(face_sel, L)
    halo_luma = mean_in(halo, L)
    separation_luma = abs(face_luma - halo_luma) / 255.0

    fa, fb = mean_in(face_sel, A), mean_in(face_sel, B)
    ha, hb = mean_in(halo, A), mean_in(halo, B)
    separation_color = float(np.hypot(fa - ha, fb - hb) / 128.0)

    face_sharp = _sharpness_masked(gray, face_sel)
    bg_sharp = _sharpness_masked(gray, background)
    blur_ratio = face_sharp / (bg_sharp + 1e-6)

    head_h = 1.35 * fh
    face_cx, face_cy = fx + fw / 2.0, fy + fh / 2.0

    def blobs(mask: np.ndarray, min_frac: float) -> list[Hotspot]:
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            frac = area / float(h * w)
            if frac < min_frac:
                continue
            cx, cy = float(cents[i][0]), float(cents[i][1])
            out.append(Hotspot(
                cx, cy, frac,
                float(np.hypot(cx - face_cx, cy - face_cy) / max(head_h, 1e-6)),
                mean_in((labels == i).astype(np.uint8) * 255, L),
            ))
        out.sort(key=lambda s: s.area_frac, reverse=True)
        return out[:6]

    # Blown or near-blown patches in the background only.
    bright = ((L > 246) & (background > 0)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    hotspots = blobs(bright, 0.0015)

    # Strongly saturated patches that outrank skin.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = ((hsv[:, :, 1] > 150) & (hsv[:, :, 2] > 90) & (background > 0)).astype(np.uint8) * 255
    sat = cv2.morphologyEx(sat, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    saturated = blobs(sat, 0.010)

    # Another recognisable face in frame is a rival for attention.
    competing = 0
    if all_faces:
        competing = sum(
            1 for f in all_faces
            if f is not face and f.eye_distance >= 0.35 * max(face.eye_distance, 1e-6)
        )

    # Lines. Vertical into the head; horizontal across the head or neck.
    band_y0, band_y1 = int(max(0, fy - 1.3 * fh)), int(min(h, fy + fh * 1.6))
    band = bgr[band_y0:band_y1, :]
    v_x = hz_y = None
    if band.size:
        edges = cv2.Canny(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), 60, 170)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 45,
                                minLineLength=int(0.45 * band.shape[0]), maxLineGap=8)
        if lines is not None and len(lines):
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                if abs(x2 - x1) <= max(3, 0.05 * w):
                    cand = (x1 + x2) / 2.0
                    if fx - 0.2 * fw <= cand <= fx + fw + 0.2 * fw:
                        v_x = float(cand)
                        break
        lines_h = cv2.HoughLinesP(edges, 1, np.pi / 180, 55,
                                  minLineLength=int(0.35 * w), maxLineGap=10)
        if lines_h is not None and len(lines_h):
            for x1, y1, x2, y2 in np.asarray(lines_h).reshape(-1, 4):
                if abs(y2 - y1) <= max(3, 0.02 * h):
                    y_abs = band_y0 + (y1 + y2) / 2.0
                    # Only counts if it crosses the head or the neck.
                    if fy - 0.4 * fh <= y_abs <= fy + fh * 1.35:
                        hz_y = float(y_abs)
                        break

    # Corner brightness — bright corners walk the eye out of the frame.
    cw, ch = max(8, w // 8), max(8, h // 8)
    corners = np.concatenate([
        L[:ch, :cw].ravel(), L[:ch, -cw:].ravel(),
        L[-ch:, :cw].ravel(), L[-ch:, -cw:].ravel(),
    ])
    corner_luma = float(corners.mean())

    bg_edges = cv2.Canny(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 60, 170)
    clutter = float(((bg_edges > 0) & (background > 0)).sum() / max((background > 0).sum(), 1))

    return BackgroundAnalysis(
        face_luma=face_luma,
        halo_luma=halo_luma,
        separation_luma=separation_luma,
        separation_color=separation_color,
        blur_ratio=blur_ratio,
        hotspots=hotspots,
        saturated_patches=saturated,
        competing_faces=competing,
        vertical_intrusion_x=v_x,
        horizontal_intrusion_y=hz_y,
        corner_luma=corner_luma,
        clutter=clutter,
        head_mask=head,
        halo_mask=halo,
    )
