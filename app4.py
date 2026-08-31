"""Shooti v4 — human portraits only.

The earlier versions tried to judge every photograph, and that breadth was the
problem: one rule set cannot know that a symmetric building wants centering and
a turned head wants nose room. v4 gives up all other subjects and gets specific.

Every rule here is portrait craft that would be meaningless elsewhere:
  - do not let the frame cut through a joint (measured in head-heights)
  - eye line and headroom targets that change with the crop
  - nose room in the direction the head is actually turned
  - do not shoot from below eye level into the nostrils
  - the near eye must be the sharpest thing in frame
  - nothing vertical growing out of the head
"""

from __future__ import annotations

import streamlit as st

from shooti.loader import load_bgr
from shooti.portrait.overlay import render, to_rgb
from shooti.portrait.rules import analyze_portrait
from shooti.subject import detect_faces

st.set_page_config(page_title="Shooti v4 — Portraits", page_icon="🧑", layout="wide")

ICON = {"major": "🔴", "minor": "🟡", "ok": "🟢"}
GROUPS = {
    "Framing": ("crop line", "eye line", "headroom", "nose room"),
    "The head": ("head tilt", "camera height"),
    "Technical": ("eye focus", "face exposure", "light direction", "subject vs background"),
    "Behind them": ("background intrusion", "background clutter"),
}

st.title("🧑 Shooti v4 — portraits only")
st.caption(
    "Narrowed to human portraits so the rules can be specific: joint-safe cropping, "
    "crop-dependent eye line, nose room from real head pose, eye sharpness."
)

with st.sidebar:
    st.header("Photo")
    source = st.radio("Source", ["Camera", "Upload"], horizontal=True)
    frame_file = (
        st.camera_input("Frame your shot")
        if source == "Camera"
        else st.file_uploader("Portrait", type=["jpg", "jpeg", "png", "webp"])
    )

    st.header("Overlay")
    layers = {
        "show_landmarks": st.checkbox("Face landmarks (the evidence)", True),
        "show_body": st.checkbox("Predicted body landmarks + crop line", True),
        "show_eye_line": st.checkbox("Eye line vs upper third", True),
        "show_pose": st.checkbox("Head pose", True),
    }

if frame_file is None:
    st.info("Add a portrait to get started.")
    st.stop()

bgr = load_bgr(frame_file.getvalue())
faces = detect_faces(bgr)

if not faces:
    st.error(
        "**No face found.** This version only analyses human portraits — it has no "
        "rules for anything else. If there is a face in this photo, it may be too "
        "small, too turned away, or too dark for the detector."
    )
    st.image(to_rgb(bgr), width="stretch")
    st.stop()

faces.sort(key=lambda f: f.eye_distance, reverse=True)
if len(faces) > 1:
    labels = [
        f"Face {i + 1} — {f.eye_distance:.0f}px between eyes, confidence {f.score:.2f}"
        for i, f in enumerate(faces)
    ]
    pick = st.selectbox(
        f"{len(faces)} faces found — which one is the subject?", range(len(faces)),
        format_func=lambda i: labels[i],
    )
    face = faces[pick]
else:
    face = faces[0]

with st.spinner("Measuring…"):
    verdict = analyze_portrait(bgr, face)

if verdict.human.verdict == "not-human":
    st.error(f"**Probably not a real person.** {verdict.human.note}")
elif verdict.human.verdict == "uncertain":
    st.warning(verdict.human.note)

left, right = st.columns([3, 2], gap="large")

with left:
    st.image(
        to_rgb(render(bgr, face, verdict.pose, verdict.crop, **layers)),
        width="stretch",
    )
    st.caption(
        "Amber dots are the detected landmarks — if they are not on a face, ignore "
        "everything else. Green dashed lines are predicted body landmarks; red ones "
        "are joints you should not crop through."
    )

with right:
    st.metric("Portrait score", f"{verdict.score} / 100")
    st.progress(verdict.score / 100)
    # Markdown rather than st.metric: crop names like "head and shoulders" get
    # truncated to "head and s..." in a metric's fixed-width value slot.
    c1, c2 = st.columns(2)
    c1.markdown(
        f"**Crop**  \n{verdict.crop.crop_name}  \n"
        f"<span style='color:#888;font-size:0.85em'>"
        f"{verdict.crop.heads_to_bottom:.2f} head-heights</span>",
        unsafe_allow_html=True,
    )
    if verdict.pose.ok:
        c2.markdown(
            f"**Head pose**  \nyaw {verdict.pose.yaw_deg:+.0f}°, pitch "
            f"{verdict.pose.pitch_deg:+.0f}°, roll {verdict.pose.roll_deg:+.0f}°  \n"
            f"<span style='color:#888;font-size:0.85em'>relative to the lens</span>",
            unsafe_allow_html=True,
        )
    else:
        c2.markdown(
            "**Head pose**  \nnot reliable  \n"
            f"<span style='color:#888;font-size:0.85em'>{verdict.pose.note}</span>",
            unsafe_allow_html=True,
        )

    problems = verdict.problems
    if problems:
        st.markdown(f"**{len(problems)} to fix**, {len(verdict.findings) - len(problems)} passing")
    else:
        st.success("Every portrait rule passes.")

st.divider()

for group, rules in GROUPS.items():
    items = [f for f in verdict.findings if f.rule in rules]
    if not items:
        continue
    bad = [f for f in items if f.severity != "ok"]
    st.markdown(f"### {group} {'— ' + str(len(bad)) + ' to fix' if bad else '✓'}")
    for f in items:
        if f.severity == "ok":
            continue
        with st.container(border=True):
            st.markdown(f"{ICON[f.severity]} **{f.rule}** — {f.message}")
            st.markdown(f"→ {f.action}")
    passing = [f for f in items if f.severity == "ok"]
    if passing:
        with st.expander(f"{len(passing)} passing in {group.lower()}"):
            for f in passing:
                st.markdown(f"🟢 **{f.rule}** — {f.message}")

if verdict.notes:
    st.divider()
    st.subheader("What this analysis assumes")
    for n in verdict.notes:
        st.warning(n)

with st.expander("Raw measurements"):
    st.code(
        "\n".join(
            [
                f"face box            {face.box}  confidence {face.score:.2f}",
                f"eye distance        {face.eye_distance:.1f} px",
                f"head height (est)   {verdict.crop.head_height_px:.1f} px",
                f"crown y (est)       {verdict.crop.crown_y:.1f}",
                f"frame ends at       {verdict.crop.heads_to_bottom:.3f} head-heights",
                f"crop classified as  {verdict.crop.crop_name}",
                f"joint cut           {verdict.crop.joint or 'none'}",
                f"head yaw            {verdict.pose.yaw_deg:+.1f} deg (+ = toward frame right)",
                f"head pitch          {verdict.pose.pitch_deg:+.1f} deg (+ = face angled down)",
                f"head roll           {verdict.pose.roll_deg:+.1f} deg",
                f"pose reliable       {verdict.pose.ok}  {verdict.pose.note}",
                f"human check         {verdict.human.verdict} (margin {verdict.human.margin:+.4f})",
            ]
        ),
        language="text",
    )
