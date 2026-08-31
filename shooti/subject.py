"""Find what the photo is *of*.

Two detectors, best first:
  1. YuNet (OpenCV DNN face detector, 227 KB ONNX). Gives a box, a confidence,
     and five landmarks — the two eyes and the nose tip are what we use, since
     eye level is the anchor real photographers compose against, and the nose
     offset tells us which way the head is turned.
  2. Gradient-energy saliency blob, for anything that isn't a face.

OpenCV 5 no longer ships the Haar cascades in the Python wheel, so the model
file is fetched once into models/ and cached. If it can't be fetched, face
detection degrades to the saliency path rather than crashing.

All boxes are pixel coords, (x, y, w, h).
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]
Point = tuple[float, float]

MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.7


def model_path() -> Path:
    override = os.environ.get("SHOOTI_YUNET_MODEL")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "models" / MODEL_FILENAME


def ensure_model(download: bool = True) -> Path | None:
    """Return the model path, fetching it once if needed. None if unavailable."""
    path = model_path()
    if path.exists():
        return path
    if not download:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(path)
        return path
    except (urllib.error.URLError, OSError):
        return None


@dataclass
class Face:
    box: Box
    score: float
    eye_a: Point  # leftmost eye in frame
    eye_b: Point  # rightmost eye in frame
    nose: Point
    # All five YuNet landmarks in the detector's own order:
    # (subject's right eye, subject's left eye, nose tip, right mouth, left mouth).
    # eye_a/eye_b above are these sorted by frame x, which loses the
    # which-side-is-which information that head-pose estimation needs.
    landmarks: tuple[Point, ...] = ()

    @property
    def eye_level(self) -> Point:
        return (
            (self.eye_a[0] + self.eye_b[0]) / 2.0,
            (self.eye_a[1] + self.eye_b[1]) / 2.0,
        )

    @property
    def eye_distance(self) -> float:
        return float(np.hypot(self.eye_b[0] - self.eye_a[0], self.eye_b[1] - self.eye_a[1]))

    @property
    def yaw(self) -> float:
        """Head turn, as nose offset from eye midpoint over inter-eye distance.

        Positive means the nose sits right of the eye midpoint, i.e. the head is
        turned toward the right of frame. Roughly |0.15|+ reads as a real turn.
        """
        ex, _ = self.eye_level
        d = self.eye_distance
        if d < 1e-6:
            return 0.0
        return float((self.nose[0] - ex) / d)

    @property
    def head_top(self) -> float:
        """Estimated top of the head — the face box starts near the brow line."""
        return self.box[1] - 0.35 * self.box[3]


@dataclass
class Subject:
    box: Box
    kind: str  # "face" | "faces" | "salient" | "none"
    confidence: float
    face: Face | None = None
    face_count: int = 0
    area_fraction: float = 0.0

    @property
    def has_subject(self) -> bool:
        return self.kind != "none"

    @property
    def center(self) -> Point:
        x, y, w, h = self.box
        return x + w / 2.0, y + h / 2.0

    @property
    def anchor(self) -> Point:
        """The point composition should be measured against.

        For people that's eye level, not the centroid of the body box — it's what
        the viewer looks at and what the thirds rule is actually about.
        """
        return self.face.eye_level if self.face else self.center

    @property
    def top(self) -> float:
        """Top of the subject, using the estimated head top when we have a face."""
        return self.face.head_top if self.face else float(self.box[1])

    def with_area_fraction(self, frame_area: float) -> "Subject":
        _, _, w, h = self.box
        self.area_fraction = (w * h) / max(frame_area, 1.0)
        return self


def detect_faces(bgr: np.ndarray, download: bool = True) -> list[Face]:
    path = ensure_model(download=download)
    if path is None:
        return []

    h, w = bgr.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        str(path), "", (w, h), score_threshold=SCORE_THRESHOLD, nms_threshold=0.3, top_k=500
    )
    detector.setInputSize((w, h))
    _, raw = detector.detect(bgr)
    if raw is None:
        return []

    faces: list[Face] = []
    for row in raw:
        x, y, bw, bh = (float(v) for v in row[:4])
        # Landmarks are five (x, y) pairs; the first two are the eyes.
        p = [(float(row[4 + 2 * i]), float(row[5 + 2 * i])) for i in range(5)]
        eyes = sorted(p[:2], key=lambda q: q[0])
        faces.append(
            Face(
                box=(int(x), int(y), int(bw), int(bh)),
                score=float(row[14]),
                eye_a=eyes[0],
                eye_b=eyes[1],
                nose=p[2],
                landmarks=tuple(p),
            )
        )
    return faces


MAX_SPAN = 0.85  # a "subject" spanning nearly the whole frame isn't a subject


def salient_region(bgr: np.ndarray, grid: int = 64) -> tuple[Box, float] | None:
    """Highest-energy compact blob of image gradient — a stand-in for "the subject".

    Returns the box plus a 0-1 confidence from how concentrated that blob's
    energy is, or None when the frame has no usable structure (a flat wall, an
    evenly textured field) — in which case claiming a subject would be a lie.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    energy = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    small = cv2.GaussianBlur(
        cv2.resize(energy, (grid, grid), interpolation=cv2.INTER_AREA), (5, 5), 0
    )

    peak = float(small.max())
    total = float(small.sum())
    if peak <= 1e-6 or total <= 1e-6:
        return None

    # Threshold on a fraction of the peak, not a percentile: in a mostly-flat
    # frame the 80th percentile is 0 and every pixel clears it.
    thresh = max(float(np.percentile(small, 85)), 0.25 * peak)
    mask = (small > thresh).astype(np.uint8)
    if mask.sum() == 0:
        return None

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    fx, fy = w / grid, h / grid

    best_box: Box | None = None
    best_energy = 0.0
    for i in range(1, count):  # row 0 is the background
        sx, sy, sw, sh = (int(v) for v in stats[i][:4])
        # Skip frame-spanning structures — a horizon line or a busy background
        # is not what the photo is of.
        if sw / grid > MAX_SPAN or sh / grid > MAX_SPAN:
            continue
        blob_energy = float(small[labels == i].sum())
        if blob_energy > best_energy:
            best_energy = blob_energy
            best_box = (
                int(sx * fx),
                int(sy * fy),
                max(1, int(sw * fx)),
                max(1, int(sh * fy)),
            )

    if best_box is None:
        # Everything that cleared the threshold spans the frame. Fall back to the
        # energy centroid with a box sized by its spread — weak, but honest.
        ys, xs = np.nonzero(mask)
        wgt = small[ys, xs]
        cx = float((xs * wgt).sum() / wgt.sum())
        cy = float((ys * wgt).sum() / wgt.sum())
        sx_ = float(np.sqrt((wgt * (xs - cx) ** 2).sum() / wgt.sum()))
        sy_ = float(np.sqrt((wgt * (ys - cy) ** 2).sum() / wgt.sum()))
        bw, bh = max(2.0, 2 * sx_), max(2.0, 2 * sy_)
        best_box = (
            int(max(0, (cx - sx_) * fx)),
            int(max(0, (cy - sy_) * fy)),
            int(min(w, bw * fx)),
            int(min(h, bh * fy)),
        )
        return best_box, 0.15

    return best_box, min(1.0, (best_energy / total) * 2.0)


def _union(boxes: list[Box]) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _intersects(a: Box, b: Box) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def _area(box: Box) -> int:
    return box[2] * box[3]


def _clamp(box: Box, w: int, h: int) -> Box:
    x = max(0, box[0])
    y = max(0, box[1])
    return x, y, min(w, box[0] + box[2]) - x, min(h, box[1] + box[3]) - y


def detect_subject(bgr: np.ndarray, download_model: bool = True) -> Subject:
    h, w = bgr.shape[:2]
    faces = detect_faces(bgr, download=download_model)

    if faces:
        faces.sort(key=lambda f: f.box[2] * f.box[3], reverse=True)
        primary = faces[0]
        if len(faces) > 1:
            box = _union([f.box for f in faces])
            subject = Subject(box, "faces", primary.score, primary, len(faces))
        else:
            # Frame the head-and-shoulders region the face implies, so headroom
            # and subject size are measured against the visible person rather
            # than a tight face crop.
            fx, fy, fw, fh = primary.box
            implied = _clamp(
                (
                    int(fx - 0.6 * fw),
                    int(fy - 0.5 * fh),
                    int(2.2 * fw),
                    int(2.9 * fh),
                ),
                w,
                h,
            )
            # A face-implied box stops at the shoulders, so a full-body shot would
            # be scored as a tiny subject. If the salient blob overlaps the face
            # and is bigger, it's most likely the body — take the union so size
            # and edge checks see the whole person.
            found = salient_region(bgr)
            if found is not None:
                blob, _ = found
                if _intersects(blob, primary.box) and _area(blob) > _area(implied):
                    implied = _clamp(_union([implied, blob]), w, h)
            subject = Subject(implied, "face", primary.score, primary, 1)
    else:
        found = salient_region(bgr)
        if found is None:
            subject = Subject((0, 0, w, h), "none", 0.0)
        else:
            box, conf = found
            subject = Subject(box, "salient", conf)

    return subject.with_area_fraction(float(h * w))
