"""Where the frame cuts the body, measured in head-heights.

The most-cited rule in portrait photography is: do not crop at a joint. A frame
that ends at the ankle, wrist, elbow or knee reads as an amputation; the same
frame moved an inch reads as a deliberate crop. Encoding that needs to know
where the joints are, and a single photo gives no body landmarks.

The classical solution is the one figure drawing uses: the head is the unit. An
adult standing figure is about 7.5 heads tall, and every landmark sits at a
known depth in head units. Face detection gives the head, so everything else
follows.

Assumptions, which the caller must surface rather than hide:
  - the subject is upright and the body extends downward from the head
  - roughly adult proportions (children run 5-6 heads, so crop names shift)
  - one subject
A seated, reclining or heavily foreshortened subject breaks all of this.
"""

from __future__ import annotations

from dataclasses import dataclass

# Depth below the crown, in head-heights. Standard 7.5-head figure canon.
LANDMARKS: dict[str, float] = {
    "crown": 0.0,
    "eyes": 0.5,
    "chin": 1.0,
    "shoulders": 1.4,
    "chest": 1.8,
    "waist": 2.7,
    "elbows": 2.85,
    "hips": 3.75,
    "wrists": 3.85,
    "mid-thigh": 4.6,
    "knees": 5.4,
    "mid-calf": 6.2,
    "ankles": 7.15,
    "feet": 7.5,
}

# Cropping inside one of these reads as severed. (low, high, joint name)
JOINT_ZONES: tuple[tuple[float, float, str], ...] = (
    (1.02, 1.30, "the neck"),
    (1.32, 1.52, "the shoulder"),
    (2.72, 3.00, "the elbow"),
    (3.68, 3.98, "the wrist"),
    (5.18, 5.62, "the knee"),
    (7.02, 7.32, "the ankle"),
)

# Where a crop line is conventionally safe. (low, high, label)
SAFE_ZONES: tuple[tuple[float, float, str], ...] = (
    (0.0, 1.0, "a face crop"),
    (1.55, 2.65, "a chest or bust crop"),
    (3.05, 3.62, "a hip crop"),
    (4.05, 5.10, "a mid-thigh crop"),
    (5.70, 6.95, "a mid-calf crop"),
    (7.40, 99.0, "a full-length crop"),
)

CROP_NAMES: tuple[tuple[float, str], ...] = (
    (0.75, "extreme close-up"),
    (1.05, "close-up (face)"),
    (1.55, "head and shoulders"),
    (2.70, "bust / chest"),
    (3.05, "waist-up"),
    (3.95, "hip / three-quarter"),
    (5.15, "mid-thigh"),
    (5.70, "knee-length"),
    (7.00, "calf-length"),
    (99.0, "full length"),
)


@dataclass
class CropGeometry:
    head_height_px: float
    crown_y: float
    heads_to_bottom: float  # where the frame's bottom edge falls, in heads
    crop_name: str
    joint: str | None  # the joint being cut, if any
    safe_zone: str | None  # the conventional crop it matches, if any
    body_visible: bool  # False for a face-only crop where the rule is moot

    @property
    def clean(self) -> bool:
        return self.joint is None


def head_height(face_box_height: float) -> float:
    """Crown-to-chin height from the detector box, which starts near the brow."""
    return 1.35 * float(face_box_height)


def crop_geometry(face_box: tuple[int, int, int, int], frame_height: int) -> CropGeometry:
    _, fy, _, fh = face_box
    hh = head_height(fh)
    crown = fy - 0.35 * fh
    heads = (frame_height - crown) / max(hh, 1e-6)

    name = next(label for limit, label in CROP_NAMES if heads < limit)
    joint = next((n for lo, hi, n in JOINT_ZONES if lo <= heads <= hi), None)
    safe = next((n for lo, hi, n in SAFE_ZONES if lo <= heads <= hi), None)
    return CropGeometry(hh, crown, heads, name, joint, safe, heads > 1.3)


def landmark_y(geo: CropGeometry, name: str) -> float:
    """Pixel row where a body landmark is predicted to fall."""
    return geo.crown_y + LANDMARKS[name] * geo.head_height_px


def nearest_safe(heads: float) -> tuple[float, str]:
    """The closest conventionally safe crop depth, for the fix suggestion."""
    best, label, dist = heads, "", float("inf")
    for lo, hi, name in SAFE_ZONES:
        for edge in (lo, hi):
            if abs(edge - heads) < dist:
                dist, best, label = abs(edge - heads), edge, name
    return best, label
