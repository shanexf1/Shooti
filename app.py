"""Shooti — Streamlit front end.

Shoot or upload, see the geometry drawn on the frame, then optionally ask Claude
for the three moves worth making.
"""

from __future__ import annotations

import streamlit as st

from shooti import analyze
from shooti.coach import CoachError, coach, measurements_text
from shooti.loader import load_bgr
from shooti.overlay import render, to_rgb

st.set_page_config(page_title="Shooti", page_icon="📷", layout="wide")

SEVERITY_ICON = {"major": "🔴", "minor": "🟡", "ok": "🟢"}

st.title("📷 Shooti")
st.caption("Composition and camera-angle coaching, measured before it's described.")

with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["Camera", "Upload"], horizontal=True)
    frame_file = (
        st.camera_input("Frame your shot")
        if source == "Camera"
        else st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"])
    )

    st.header("Overlay")
    layers = {
        "grid": st.checkbox("Rule-of-thirds grid", True),
        "subject": st.checkbox("Subject box", True),
        "target": st.checkbox("Target crosshair", True),
        "arrow": st.checkbox("Move arrow", True),
        "horizon": st.checkbox("Horizon / level", True),
    }

    st.header("AI coaching")
    intent = st.text_input("What are you going for?", placeholder="moody portrait at dusk")
    api_key = st.text_input("Anthropic API key", type="password", help="Leave blank to use ANTHROPIC_API_KEY or an `ant auth login` profile.")
    want_coaching = st.button("Ask Claude for moves", width="stretch")

if frame_file is None:
    st.info("Take a photo or upload one to get started.")
    st.stop()

bgr = load_bgr(frame_file.getvalue())
analysis = analyze(bgr)
overlay = render(bgr, analysis, **layers)

left, right = st.columns([3, 2], gap="large")

with left:
    st.image(to_rgb(overlay), width="stretch")
    st.caption(
        "Cyan = what Shooti thinks the subject is. Magenta crosshair = where its "
        "center should sit. Red horizon line = off level."
    )

with right:
    score = analysis.score
    st.metric("Composition score", f"{score}/100")
    st.progress(score / 100)

    problems = analysis.problems
    if not problems:
        st.success("Nothing to fix — this framing holds up on every rule checked.")
    else:
        st.subheader(f"{len(problems)} thing{'s' if len(problems) > 1 else ''} to fix")
        for finding in problems:
            with st.container(border=True):
                st.markdown(f"{SEVERITY_ICON[finding.severity]} **{finding.rule}** — {finding.message}")
                st.markdown(f"→ {finding.action}")

    with st.expander("Rules that passed"):
        for finding in analysis.findings:
            if finding.severity == "ok":
                st.markdown(f"🟢 **{finding.rule}** — {finding.message}")

st.divider()

if want_coaching:
    with st.spinner("Asking Claude…"):
        try:
            result = coach(bgr, analysis, api_key=api_key or None, intent=intent or None)
        except CoachError as exc:
            st.error(str(exc))
        else:
            st.subheader("Claude's take")
            st.markdown(result.text)
            st.caption(
                f"{result.model} · {result.input_tokens} in / {result.output_tokens} out tokens"
            )
else:
    st.caption("The score and fixes above are pure computer vision — no API key needed. "
               "Claude adds scene-aware prioritization on request.")

with st.expander("What the CV layer measured"):
    st.code(measurements_text(analysis), language="text")
