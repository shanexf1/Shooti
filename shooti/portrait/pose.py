"""Head pose from the five face landmarks, via solvePnP.

Portrait craft is mostly about the head's orientation relative to the lens —
whether you are shooting up someone's nose, whether the head is turned enough to
need nose room, whether the tilt is a flattering slight cant or a mistake. None
of that is answerable from a bounding box, so this recovers actual angles.

Method: fit a canonical 3D face to the five detected landmarks. Five points is
enough because the nose tip sits off the plane of the eyes and mouth, which is
exactly what makes the pose observable.

An honest ambiguity, stated once and carried into the advice: a single face
cannot distinguish "the camera is above the subject" from "the subject tilted
their head down". Both produce the same landmark geometry. So pitch is reported
as the face's angle RELATIVE TO THE LENS, which is what actually affects the
photograph, and the advice offers both remedies.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..subject import Face

# Canonical face, millimetres. Axes: X right, Y down, Z away from camera.
# Nose tip at the origin, so it is the point nearest the lens.
MODEL_POINTS = np.array(
    [
        (-32.0, -32.0, 28.0),  # subject's right eye centre
        (32.0, -32.0, 28.0),  # subject's left eye centre
        (0.0, 0.0, 0.0),  # nose tip
        (-26.0, 30.0, 22.0),  # right mouth corner
        (26.0, 30.0, 22.0),  # left mouth corner
    ],
    dtype=np.float64,
)


@dataclass
class HeadPose:
    yaw_deg: float  # + = face turned toward the RIGHT of frame
    pitch_deg: float  # + = face angled DOWN relative to the lens
    roll_deg: float  # + = head tilted so the subject's left ear drops
    ok: bool
    note: str = ""

    @property
    def turned(self) -> bool:
        return abs(self.yaw_deg) >= 12.0

    @property
    def near_frontal(self) -> bool:
        return abs(self.yaw_deg) < 12.0 and abs(self.pitch_deg) < 8.0


def roll_from_eyes(face: Face) -> float:
    """Roll straight from the eye line — more robust than recovering it from PnP.

    Positive means the eye line descends toward the right of frame, i.e. the head
    is canted with the subject's left side dropping.
    """
    (ax, ay), (bx, by) = face.eye_a, face.eye_b  # already sorted left-to-right
    return float(np.degrees(np.arctan2(by - ay, bx - ax)))


def estimate(face: Face, frame_shape: tuple[int, int]) -> HeadPose:
    """Recover head pose. Returns ok=False with a reason when it can't."""
    roll = roll_from_eyes(face)

    if len(face.landmarks) != 5:
        return HeadPose(0.0, 0.0, roll, False, "Detector returned no landmarks.")

    h, w = frame_shape[:2]
    # Focal length is unknown, so assume a normal-ish lens: f = image width.
    # Pose is therefore approximate, and more so for heavily cropped frames.
    focal = float(w)
    camera = np.array(
        [[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64
    )
    image_points = np.array(face.landmarks, dtype=np.float64)

    # A face smaller than a few dozen pixels gives landmark noise the same
    # magnitude as the geometry we are trying to measure.
    if face.eye_distance < 12.0:
        return HeadPose(0.0, 0.0, roll, False,
                        f"Face is too small ({face.eye_distance:.0f}px between the eyes) "
                        "for a reliable pose estimate.")

    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return HeadPose(0.0, 0.0, roll, False, "Pose fit did not converge.")

    rot, _ = cv2.Rodrigues(rvec)
    # Where the nose points, in camera space. In model coords the nose axis is
    # -Z (toward the lens), so a front-facing head gives forward ~ (0, 0, -1).
    forward = rot @ np.array([0.0, 0.0, -1.0])
    fx, fy, fz = (float(v) for v in forward)

    yaw = float(np.degrees(np.arctan2(fx, -fz)))
    pitch = float(np.degrees(np.arcsin(np.clip(fy, -1.0, 1.0))))
    return HeadPose(yaw, pitch, roll, True)


def describe(pose: HeadPose) -> str:
    if not pose.ok:
        return pose.note
    bits = []
    if abs(pose.yaw_deg) < 6:
        bits.append("facing the lens")
    else:
        bits.append(f"turned {abs(pose.yaw_deg):.0f}° toward frame "
                    f"{'right' if pose.yaw_deg > 0 else 'left'}")
    if abs(pose.pitch_deg) >= 6:
        bits.append(f"angled {abs(pose.pitch_deg):.0f}° "
                    f"{'down' if pose.pitch_deg > 0 else 'up'} relative to the lens")
    if abs(pose.roll_deg) >= 3:
        bits.append(f"tilted {abs(pose.roll_deg):.0f}°")
    return ", ".join(bits)
