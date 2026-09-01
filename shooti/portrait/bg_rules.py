"""Background findings for portraits.

Separate from rules.py so v4 keeps exactly the rule set it was validated with,
and v4.1 adds these on top.

One nuance that matters: a large, even, blown-out region is a deliberate white
backdrop, not a distracting hotspot. Without that exception every studio portrait
gets scolded for its own background — the first run flagged three "hotspots" on a
plain white studio shot.
"""

from __future__ import annotations

from ..rules import Finding
from ..subject import Face
from .background import BackgroundAnalysis
from .styles import NEUTRAL, Style

SEPARATION_LOW = 0.10
BLUR_WANTED = 1.5
CLUTTER_HIGH = 0.14
HOTSPOT_AREA = 0.004
HOTSPOT_NEAR = 2.5  # head-heights from the face
BACKDROP_AREA = 0.18  # a bright region this large is probably intentional
BACKDROP_CLUTTER = 0.06
CORNER_PULL = 40.0
SATURATED_AREA = 0.02


def _f(rule, sev, msg, action, penalty=0.0, **data) -> Finding:
    return Finding(rule, sev, msg, action, penalty=penalty, data=data)


def _separation(b: BackgroundAnalysis, st: Style = NEUTRAL) -> Finding:
    if b.merges:
        return _f("subject separation", "major",
                  f"The subject is dissolving into the background — tone behind the head is "
                  f"{b.halo_luma:.0f} against {b.face_luma:.0f} on the face, and the colours "
                  "match too. The silhouette has nothing to read against.",
                  "Move them so a lighter or darker area sits behind the head, add a hair or "
                  "rim light, or change your angle so the background behind them changes.",
                  penalty=18.0, sep_luma=b.separation_luma, sep_color=b.separation_color)
    if b.separation_luma < st.separation_low and b.separation_color >= st.separation_low:
        return _f("subject separation", "ok",
                  "Tonally the subject and background are close, but the colours differ enough "
                  "to hold them apart.",
                  "Nothing to correct.", sep_color=b.separation_color)
    if b.separation_luma < st.separation_low:
        return _f("subject separation", "minor",
                  f"Little tonal contrast behind the head ({b.halo_luma:.0f} vs "
                  f"{b.face_luma:.0f} on the face).",
                  "Shift a step so a different tone falls behind them, or light the background "
                  "separately.",
                  penalty=8.0, sep_luma=b.separation_luma)
    return _f("subject separation", "ok",
              f"The head reads clearly against its background ({b.face_luma:.0f} on the face vs "
              f"{b.halo_luma:.0f} behind it).",
              "Nothing to correct.", sep_luma=b.separation_luma)


def _depth(b: BackgroundAnalysis, st: Style = NEUTRAL) -> Finding:
    if b.blur_ratio >= st.blur_wanted:
        return _f("background blur", "ok",
                  f"The face is {b.blur_ratio:.1f}× sharper than the background, so the "
                  "background stays subordinate.",
                  "Nothing to correct.", blur_ratio=b.blur_ratio)
    if b.blur_ratio < 1.0:
        return _f("background blur", "major",
                  f"The background is sharper than the face ({b.blur_ratio:.2f}× — under 1.0 "
                  "means the background won). It competes directly with the subject.",
                  "Open the aperture, move the subject further from the background, or step back "
                  "and use a longer focal length. Check focus landed on the near eye.",
                  penalty=16.0, blur_ratio=b.blur_ratio)
    return _f("background blur", "minor",
              f"The background is nearly as sharp as the face ({b.blur_ratio:.2f}×), so it "
              "holds detail that pulls attention.",
              "Open up a stop or two, or put more distance between subject and background.",
              penalty=min(10.0, (st.blur_wanted - b.blur_ratio) * 18.0), blur_ratio=b.blur_ratio)


def _hotspots(b: BackgroundAnalysis) -> Finding:
    if not b.hotspots:
        return _f("bright distractions", "ok", "No blown-out patches behind the subject.",
                  "Nothing to correct.")
    biggest = b.hotspots[0]
    total = sum(s.area_frac for s in b.hotspots)

    # A large, even, blown region is a backdrop the photographer chose.
    if total >= BACKDROP_AREA and b.clutter <= BACKDROP_CLUTTER:
        return _f("bright distractions", "ok",
                  f"The bright area behind the subject covers {total * 100:.0f}% of the frame and "
                  "is even — that reads as an intentional white backdrop, not a distraction.",
                  "Nothing to correct.", backdrop=True)

    near = [s for s in b.hotspots if s.distance_heads <= HOTSPOT_NEAR and s.area_frac >= HOTSPOT_AREA]
    if not near:
        return _f("bright distractions", "minor",
                  f"{len(b.hotspots)} blown patch(es) in the background, but well away from the "
                  "head.",
                  "Worth cloning out later, or crop them off.",
                  penalty=3.0, count=len(b.hotspots))
    return _f("bright distractions", "major" if biggest.area_frac > 0.02 else "minor",
              f"A blown-out patch sits {near[0].distance_heads:.1f} head-heights from the face "
              f"and covers {near[0].area_frac * 100:.1f}% of the frame. The eye goes to the "
              "brightest thing, and right now that is not the face.",
              "Step aside so it falls outside the frame, drop your angle, or shade it. If it is a "
              "window or sky gap, put the subject where it sits behind them, not beside them.",
              penalty=min(15.0, 6.0 + near[0].area_frac * 260.0),
              count=len(near))


def _rivals(b: BackgroundAnalysis) -> Finding:
    if b.competing_faces == 0:
        return _f("competing faces", "ok", "No other faces competing for attention.",
                  "Nothing to correct.")
    return _f("competing faces", "major" if b.competing_faces > 1 else "minor",
              f"{b.competing_faces} other recognisable face(s) in the frame. A viewer looks at "
              "every face, so a second one splits the portrait.",
              "Reframe to exclude them, wait for them to leave, or open the aperture so they "
              "blur beyond recognition.",
              penalty=6.0 * b.competing_faces, count=b.competing_faces)


def _lines(b: BackgroundAnalysis, face: Face) -> list[Finding]:
    out: list[Finding] = []
    if b.vertical_intrusion_x is not None:
        out.append(_f("vertical intrusion", "major",
                      "A strong vertical runs down into the head — a pole, post, trunk or door "
                      "frame appearing to grow out of the subject.",
                      "Step left or right a pace, or lower the camera, so it clears the head.",
                      penalty=14.0))
    else:
        out.append(_f("vertical intrusion", "ok", "Nothing vertical growing out of the head.",
                      "Nothing to correct."))

    if b.horizontal_intrusion_y is not None:
        fy, fh = face.box[1], face.box[3]
        y = b.horizontal_intrusion_y
        where = "the head" if y <= fy + fh else "the neck"
        out.append(_f("horizontal intrusion", "major" if where == "the neck" else "minor",
                      f"A long horizontal line — a horizon, wall junction, table or railing — "
                      f"crosses {where}. It reads as slicing through the subject.",
                      f"Raise or lower the camera so the line falls clear of the head and neck, "
                      "or move the subject up or down against it.",
                      penalty=12.0 if where == "the neck" else 7.0))
    else:
        out.append(_f("horizontal intrusion", "ok",
                      "No horizontal line cutting across the head or neck.", "Nothing to correct."))
    return out


def _colour(b: BackgroundAnalysis) -> Finding:
    strong = [s for s in b.saturated_patches if s.area_frac >= SATURATED_AREA]
    if not strong:
        return _f("colour distraction", "ok",
                  "No strongly saturated area in the background outranking the skin.",
                  "Nothing to correct.")
    biggest = strong[0]
    return _f("colour distraction", "minor",
              f"A strongly saturated area covers {biggest.area_frac * 100:.0f}% of the frame "
              f"{biggest.distance_heads:.1f} head-heights from the face. Saturated colour "
              "competes with skin, which is comparatively muted.",
              "Reframe it out, or desaturate it in post. Failing that, put the subject in front "
              "of it rather than beside it.",
              penalty=min(10.0, biggest.area_frac * 90.0), count=len(strong))


def _escape(b: BackgroundAnalysis) -> Finding:
    pull = b.corner_luma - b.face_luma
    if pull <= CORNER_PULL:
        return _f("eye escape", "ok",
                  f"The corners ({b.corner_luma:.0f}) do not outshine the face ({b.face_luma:.0f}).",
                  "Nothing to correct.", pull=pull)
    return _f("eye escape", "minor",
              f"The frame corners are much brighter than the face ({b.corner_luma:.0f} vs "
              f"{b.face_luma:.0f}), which walks the eye out of the picture.",
              "Crop tighter, or darken the corners slightly in post so the face stays the "
              "brightest thing.",
              penalty=min(9.0, (pull - CORNER_PULL) * 0.16), pull=pull)


def _clutter(b: BackgroundAnalysis, st: Style = NEUTRAL) -> Finding:
    if b.clutter <= st.clutter_high:
        return _f("background clutter", "ok",
                  f"The background is reasonably clean (edge density {b.clutter:.2f}).",
                  "Nothing to correct.", clutter=b.clutter)
    return _f("background clutter", "major" if b.clutter > st.clutter_high * 1.7 else "minor",
              f"The background is busy (edge density {b.clutter:.2f}) and competes with the face.",
              "Open the aperture, move the subject away from the background, find a plainer wall, "
              "or shoot from a lower or higher angle to put sky or ground behind them.",
              penalty=min(14.0, (b.clutter - st.clutter_high) * 55.0), clutter=b.clutter)


def evaluate(b: BackgroundAnalysis, face: Face, style: Style = NEUTRAL) -> list[Finding]:
    findings = [
        _separation(b, style),
        _depth(b, style),
        _hotspots(b),
        _rivals(b),
        _colour(b),
        _escape(b),
        _clutter(b, style),
    ]
    findings += _lines(b, face)
    return findings
