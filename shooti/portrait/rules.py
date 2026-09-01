"""Portrait rules. Narrow domain, specific rules.

This is the payoff of restricting to human portraits: every rule here would be
meaningless or wrong for a landscape. "Do not crop at the knee", "the near eye
must be sharp", "a turned head needs nose room", "do not shoot up the nostrils"
are not general composition principles — they are portrait craft.

Targets vary by crop. A close-up wants the eyes near 40% down and almost no
headroom; a full-length shot wants the eyes near 20% and real air above. v1
applied one headroom range to every photograph, which is how it managed to be
wrong about everything at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..rules import Finding
from ..subject import Face
from .anatomy import CropGeometry, crop_geometry, nearest_safe
from .human import HumanCheck
from .pose import HeadPose
from .quality import EyeFocus, FaceLight, HeadBackground
from .styles import NEUTRAL, Style

# (eye-line low, high), (headroom low, high) as fractions of frame height,
# keyed by the crop name from anatomy.CROP_NAMES.
CROP_TARGETS: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "extreme close-up": ((0.30, 0.50), (-0.10, 0.04)),
    "close-up (face)": ((0.28, 0.45), (0.00, 0.07)),
    "head and shoulders": ((0.26, 0.42), (0.02, 0.11)),
    "bust / chest": ((0.22, 0.40), (0.04, 0.14)),
    "waist-up": ((0.18, 0.36), (0.05, 0.16)),
    "hip / three-quarter": ((0.15, 0.33), (0.05, 0.17)),
    "mid-thigh": ((0.13, 0.30), (0.06, 0.18)),
    "knee-length": ((0.12, 0.28), (0.06, 0.18)),
    "calf-length": ((0.10, 0.26), (0.06, 0.19)),
    "full length": ((0.08, 0.24), (0.06, 0.20)),
}

TILT_FLATTERING = (3.0, 16.0)
TILT_EXCESSIVE = 22.0
PITCH_UP_LIMIT = -9.0  # face angled up relative to lens = shooting from below
PITCH_DOWN_LIMIT = 16.0
YAW_TURNED = 12.0
NOSE_ROOM_MIN = 0.55
EYE_FOCUS_MIN = 1.15
CLUTTER_HIGH = 0.16


@dataclass
class PortraitVerdict:
    crop: CropGeometry
    pose: HeadPose
    human: HumanCheck
    findings: list[Finding] = field(default_factory=list)
    score: int = 100
    notes: list[str] = field(default_factory=list)
    background: object | None = None  # BackgroundAnalysis when v4.1 ran
    style: object | None = None  # Style actually applied

    @property
    def problems(self) -> list[Finding]:
        return [f for f in self.findings if f.severity != "ok"]


def _f(rule, sev, msg, action, penalty=0.0, **data) -> Finding:
    return Finding(rule, sev, msg, action, penalty=penalty, data=data)


def _crop_line(geo: CropGeometry, st: Style = NEUTRAL) -> Finding:
    if not geo.body_visible:
        return _f("crop line", "ok",
                  f"A {geo.crop_name} — the frame ends above the body, so there's no joint to cut.",
                  "Nothing to correct.")
    if geo.joint:
        target, label = nearest_safe(geo.heads_to_bottom)
        delta = target - geo.heads_to_bottom
        direction = "wider (step back or zoom out)" if delta > 0 else "tighter (step in or zoom in)"
        return _f("crop line", st.crop_joint_severity,
                  f"The frame cuts through {geo.joint}, at {geo.heads_to_bottom:.2f} head-heights. "
                  "A crop landing on a joint reads as an amputation.",
                  f"Reframe about {abs(delta):.2f} head-heights {direction} to land on {label}.",
                  penalty=18.0 if st.crop_joint_severity == "major" else 8.0,
                  heads=geo.heads_to_bottom, joint=geo.joint)
    if geo.safe_zone:
        return _f("crop line", "ok",
                  f"The frame ends at {geo.heads_to_bottom:.2f} head-heights — {geo.safe_zone}, "
                  "clear of every joint.",
                  "Hold this crop.", heads=geo.heads_to_bottom)
    return _f("crop line", "minor",
              f"The frame ends at {geo.heads_to_bottom:.2f} head-heights, between the conventional crops.",
              "It misses the joints, but tightening or widening slightly would hit a standard crop.",
              penalty=4.0, heads=geo.heads_to_bottom)


def _eye_line(face: Face, geo: CropGeometry, frame_h: int) -> Finding:
    low, high = CROP_TARGETS.get(geo.crop_name, ((0.25, 0.40), (0.03, 0.14)))[0]
    _, ey = face.eye_level
    frac = ey / frame_h
    if low <= frac <= high:
        return _f("eye line", "ok",
                  f"Eyes sit {frac * 100:.0f}% down the frame — in range for a {geo.crop_name}.",
                  "Hold this height.", frac=frac)
    if frac < low:
        delta = (low - frac) * frame_h
        return _f("eye line", "minor" if low - frac < 0.10 else "major",
                  f"Eyes sit {frac * 100:.0f}% down; a {geo.crop_name} wants "
                  f"{low * 100:.0f}-{high * 100:.0f}%. They read too high, which crowds the top.",
                  f"Raise the camera about {delta / frame_h * 100:.0f}% of the frame height, "
                  "so the subject drops in frame.",
                  penalty=min(16.0, (low - frac) * 110.0), frac=frac)
    delta = (frac - high) * frame_h
    return _f("eye line", "minor" if frac - high < 0.10 else "major",
              f"Eyes sit {frac * 100:.0f}% down; a {geo.crop_name} wants "
              f"{low * 100:.0f}-{high * 100:.0f}%. Too low leaves dead space above.",
              f"Lower the camera about {delta / frame_h * 100:.0f}% of the frame height.",
              penalty=min(16.0, (frac - high) * 110.0), frac=frac)


def _headroom(face: Face, geo: CropGeometry, frame_h: int) -> Finding:
    low, high = CROP_TARGETS.get(geo.crop_name, ((0.25, 0.40), (0.03, 0.14)))[1]
    gap = (face.box[1] - 0.35 * face.box[3]) / frame_h
    if low <= gap <= high:
        return _f("headroom", "ok",
                  f"Headroom {gap * 100:.0f}% — right for a {geo.crop_name}.",
                  "Hold it.", gap=gap)
    if gap > high:
        return _f("headroom", "major" if gap - high > 0.10 else "minor",
                  f"{gap * 100:.0f}% of the frame is empty above the head; a {geo.crop_name} "
                  f"wants at most {high * 100:.0f}%.",
                  "Tilt down or lower the camera, or crop the top.",
                  penalty=min(18.0, (gap - high) * 95.0), gap=gap)
    if gap < 0 and geo.crop_name in ("extreme close-up", "close-up (face)"):
        return _f("headroom", "ok",
                  "The crown is cropped, which is a deliberate and normal choice this tight.",
                  "Nothing to correct.", gap=gap)
    return _f("headroom", "minor",
              f"Headroom is {gap * 100:.0f}% — tight for a {geo.crop_name}; the head crowds the edge.",
              "Tilt up slightly or step back.", penalty=8.0, gap=gap)


def _nose_room(face: Face, pose: HeadPose, frame_w: int, st: Style = NEUTRAL) -> Finding:
    if not pose.ok:
        return _f("nose room", "ok", "Head pose isn't reliable here, so gaze direction is unknown.",
                  "Nothing to correct.")
    if abs(pose.yaw_deg) < YAW_TURNED:
        return _f("nose room", "ok",
                  f"Head is within {YAW_TURNED:.0f}° of facing the lens, so side space can stay even.",
                  "Nothing to correct.", yaw=pose.yaw_deg)
    ex, _ = face.eye_level
    nx = ex / frame_w
    ahead = (1.0 - nx) if pose.yaw_deg > 0 else nx
    facing = "right" if pose.yaw_deg > 0 else "left"
    if ahead >= st.nose_room_min:
        return _f("nose room", "ok",
                  f"Turned {abs(pose.yaw_deg):.0f}° toward frame {facing}, with "
                  f"{ahead * 100:.0f}% of the frame open that way.",
                  "Nothing to correct.", ahead=ahead)
    return _f("nose room", "major" if ahead < 0.4 else "minor",
              f"Turned {abs(pose.yaw_deg):.0f}° toward frame {facing}, but only "
              f"{ahead * 100:.0f}% of the frame is open ahead of the face. The gaze runs into the edge.",
              f"Pan {facing} to put more space in front of the face than behind the head.",
              penalty=min(16.0, (st.nose_room_min - ahead) * 70.0), ahead=ahead)


def _head_tilt(pose: HeadPose, st: Style = NEUTRAL) -> Finding:
    roll = abs(pose.roll_deg)
    if roll > st.tilt_excessive:
        return _f("head tilt", "minor",
                  f"Head is canted {roll:.0f}°, past the point where it reads as a choice.",
                  "Straighten the head, or commit and level the eye line to the frame.",
                  penalty=8.0, roll=pose.roll_deg)
    if TILT_FLATTERING[0] <= roll <= TILT_FLATTERING[1]:
        return _f("head tilt", "ok",
                  f"Head tilted {roll:.0f}° — the slight cant that usually flatters.",
                  "Keep it.", roll=pose.roll_deg)
    return _f("head tilt", "ok",
              f"Head is level ({roll:.0f}° of tilt), which reads formal and steady.",
              "Fine as is; a 5-10° tilt would soften it if you want that.", roll=pose.roll_deg)


def _face_angle(pose: HeadPose, st: Style = NEUTRAL) -> Finding:
    if not pose.ok:
        return _f("camera height", "ok", pose.note, "Nothing to correct.")
    p = pose.pitch_deg
    if p < st.pitch_up_limit:
        return _f("camera height", "major" if p < -18 else "minor",
                  f"The face is angled {abs(p):.0f}° UP relative to the lens — you are shooting "
                  "from below eye level, which foreshortens the chin and looks up the nostrils.",
                  "Raise the camera to at least eye level, or ask them to lower their chin. "
                  "(A single photo can't separate a low camera from a raised chin — either fix works.)",
                  penalty=min(20.0, abs(p - st.pitch_up_limit) * 1.2), pitch=p)
    if p > st.pitch_down_limit:
        return _f("camera height", "minor",
                  f"The face is angled {p:.0f}° DOWN relative to the lens — a high camera or a "
                  "dropped chin. Slightly above eye level flatters; this far is a lot.",
                  "Lower the camera toward eye level, or ask them to lift their chin.",
                  penalty=min(12.0, (p - st.pitch_down_limit) * 0.8), pitch=p)
    return _f("camera height", "ok",
              f"The face is within {max(abs(p), 1):.0f}° of square to the lens — around eye level.",
              "Hold this height.", pitch=p)


def _eye_focus(focus: EyeFocus, st: Style = NEUTRAL) -> Finding:
    if focus.ratio_to_background >= st.eye_focus_min:
        return _f("eye focus", "ok",
                  f"The eyes are the sharpest thing in frame ({focus.ratio_to_background:.1f}× "
                  "the background).",
                  "Nothing to correct.", ratio=focus.ratio_to_background)
    return _f("eye focus", "major" if focus.ratio_to_background < 0.9 else "minor",
              f"The eyes are not clearly the sharpest part of the frame "
              f"({focus.ratio_to_background:.1f}× the background). In a portrait the near eye "
              "carries the picture.",
              "Focus on the near eye — single-point AF or eye-detect — and check the shutter "
              "speed is fast enough to freeze small movements.",
              penalty=min(20.0, (st.eye_focus_min - focus.ratio_to_background) * 26.0),
              ratio=focus.ratio_to_background)


def _lighting(light: FaceLight) -> list[Finding]:
    out: list[Finding] = []
    if light.clipped_highlight > 0.02:
        out.append(_f("face exposure", "major" if light.clipped_highlight > 0.06 else "minor",
                      f"{light.clipped_highlight * 100:.1f}% of the face is blown to pure white — "
                      "that detail cannot be recovered.",
                      "Reduce exposure, or move them out of direct sun into open shade.",
                      penalty=min(16.0, light.clipped_highlight * 180.0)))
    elif light.crushed_shadow > 0.12:
        out.append(_f("face exposure", "minor",
                      f"{light.crushed_shadow * 100:.0f}% of the face is crushed to black.",
                      "Add fill light or a reflector, or raise exposure and protect the highlights.",
                      penalty=min(12.0, light.crushed_shadow * 40.0)))
    else:
        out.append(_f("face exposure", "ok",
                      f"Face exposure is clean (mean {light.mean:.0f}/255, no clipping).",
                      "Nothing to correct."))

    gap = light.background_mean - light.mean
    if gap > 45:
        out.append(_f("subject vs background", "minor",
                      f"The background is much brighter than the face ({light.background_mean:.0f} "
                      f"vs {light.mean:.0f}) — the eye goes to the background first.",
                      "Expose for the face and let the background blow, or turn them so the light "
                      "falls on the face.",
                      penalty=min(12.0, (gap - 45) * 0.25)))
    else:
        out.append(_f("subject vs background", "ok",
                      f"The face holds against its background ({light.mean:.0f} vs "
                      f"{light.background_mean:.0f}).",
                      "Nothing to correct."))

    if abs(light.side_ratio) < 0.03 and light.mean > 60:
        out.append(_f("light direction", "minor",
                      "The light is dead flat across the face, which renders features without shape.",
                      "Move them, or yourself, so the light comes from 30-45° to one side.",
                      penalty=5.0))
    else:
        out.append(_f("light direction", "ok",
                      f"Light reads {light.direction}, which gives the face some modelling.",
                      "Nothing to correct."))
    return out


def _background(bg: HeadBackground) -> list[Finding]:
    out: list[Finding] = []
    if bg.intrusion:
        out.append(_f("background intrusion", "major",
                      "A strong vertical line runs down into the head — a pole, post or trunk "
                      "appearing to grow out of them.",
                      "Step left or right a pace, or drop the camera slightly, so it clears the head.",
                      penalty=14.0))
    else:
        out.append(_f("background intrusion", "ok",
                      "Nothing vertical is growing out of the head.", "Nothing to correct."))

    if bg.clutter > CLUTTER_HIGH:
        out.append(_f("background clutter", "minor",
                      f"The area behind the head is busy (edge density {bg.clutter:.2f}), which "
                      "competes with the face.",
                      "Open the aperture, move them further from the background, or find a "
                      "plainer backdrop.",
                      penalty=min(10.0, (bg.clutter - CLUTTER_HIGH) * 45.0)))
    else:
        out.append(_f("background clutter", "ok",
                      f"The area behind the head is reasonably clean ({bg.clutter:.2f}).",
                      "Nothing to correct."))
    return out


def evaluate(
    bgr: np.ndarray,
    face: Face,
    pose: HeadPose,
    human: HumanCheck,
    focus: EyeFocus,
    light: FaceLight,
    bg: HeadBackground,
    style: Style = NEUTRAL,
) -> PortraitVerdict:
    h, w = bgr.shape[:2]
    geo = crop_geometry(face.box, h)

    findings = [
        _crop_line(geo, style),
        _eye_line(face, geo, h),
        _headroom(face, geo, h),
        _nose_room(face, pose, w, style),
        _head_tilt(pose, style),
        _face_angle(pose, style),
        _eye_focus(focus, style),
    ]
    findings += _lighting(light)
    findings += _background(bg)

    notes = []
    if geo.body_visible:
        notes.append(
            "Body landmarks are predicted from head-height proportions, assuming an upright "
            "adult. A seated, reclining or foreshortened subject makes the crop-line finding "
            "unreliable — check the drawn lines against the actual body."
        )
    if not pose.ok:
        notes.append(f"Head pose unavailable: {pose.note}")
    if human.verdict == "not-human":
        notes.append(human.note)
    elif human.verdict == "uncertain":
        notes.append(human.note)

    score = int(round(max(0.0, 100.0 - sum(f.penalty for f in findings))))
    order = {"major": 0, "minor": 1, "ok": 2}
    findings.sort(key=lambda f: (order[f.severity], -f.penalty))
    v = PortraitVerdict(geo, pose, human, findings, score, notes)
    v.style = style
    return v


def analyze_portrait(
    bgr: np.ndarray,
    face: Face,
    *,
    all_faces: list[Face] | None = None,
    deep_background: bool = False,
    style: Style = NEUTRAL,
) -> PortraitVerdict:
    """Run every measurement and evaluate.

    deep_background=False keeps exactly the rule set v4 was built with.
    deep_background=True adds v4.1's background findings, which replace the two
    coarse background rules with nine specific ones.
    """
    from .human import check as human_check
    from .pose import estimate
    from .quality import eye_focus, face_light, head_background

    verdict = evaluate(
        bgr, face,
        estimate(face, bgr.shape),
        human_check(bgr, face.box),
        eye_focus(bgr, face),
        face_light(bgr, face),
        head_background(bgr, face),
        style,
    )
    if not deep_background:
        return verdict

    from . import background as bgmod
    from . import bg_rules

    analysis = bgmod.analyze(bgr, face, all_faces)
    # The v4 background rules are superseded by the detailed ones.
    kept = [f for f in verdict.findings
            if f.rule not in ("background intrusion", "background clutter")]
    verdict.findings = kept + bg_rules.evaluate(analysis, face, style)
    verdict.background = analysis
    verdict.score = int(round(max(0.0, 100.0 - sum(f.penalty for f in verdict.findings))))
    order = {"major": 0, "minor": 1, "ok": 2}
    verdict.findings.sort(key=lambda f: (order[f.severity], -f.penalty))
    return verdict
