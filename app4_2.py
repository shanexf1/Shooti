"""Shooti v4.2 — the model adjusts the judging, not just the wording.

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

Every rule above is measured in Python and needs no API key. On top of that,
optional coaching can be requested from either Claude or ChatGPT — switchable in
the sidebar — which is given the photo plus these measurements and told not to
re-estimate the geometry itself.

v4.2 answers a fair criticism of v4 and v4.1: the LLM was downstream of judging.
It read the findings and narrated them, and could not change the verdict. Every
threshold was a hard-coded constant.

Now the model does two things that change the verdict:

  1. Picks the STYLE, which shifts every tolerance. A documentary frame is not
     judged like a passport photo — candid work tolerates a cant, a busy
     background and a background as sharp as the face.
  2. Reviews each finding and may DISMISS it (the measurement is wrong about this
     photo) or ESCALATE it (worse than measured), with a stated reason.

The measurements have blind spots the model can see straight past: "vertical
intrusion" finds a line, not a pole, so it cannot tell a lamp post from the
subject's own raised arm.

Guarded, because a model that dismisses everything to be agreeable would be
worse than no review: at most 3 dismissals, each needs a specific reason,
dismissals may not lift the score by more than 25 points, and every decision is
shown with its reason. The raw verdict stays visible alongside.

v4.1's background analysis is retained. v4 had two coarse background
rules. This replaces them with nine, measured in a ring immediately around the
head rather than across the whole frame, because that is where the eye actually
compares:

  subject separation   is the silhouette dissolving into what is behind it
  background blur      is the background subordinate, or competing
  bright distractions  blown patches near the face (with a white-backdrop exception)
  competing faces      another face the viewer will look at
  colour distraction   saturated areas that outrank skin
  eye escape           corners brighter than the face
  background clutter   general busy-ness
  vertical intrusion   a pole growing out of the head
  horizontal intrusion a line slicing the neck
"""

from __future__ import annotations

import hashlib

import streamlit as st

from shooti.loader import load_bgr
from shooti.portrait.coach import (
    DEFAULT_MODELS,
    KEY_ENV,
    PROVIDER_LABELS,
    CoachError,
    coach,
    list_models,
    measurements_text,
)
from shooti.portrait.overlay import render, to_rgb
from shooti.portrait.adjudicate import MAX_DISMISS, MAX_SCORE_GAIN, adjudicate
from shooti.portrait.rules import analyze_portrait
from shooti.portrait.styles import STYLES
from shooti.subject import detect_faces

st.set_page_config(page_title="Shooti v4.2 — Portraits", page_icon="⚖️", layout="wide")

ICON = {"major": "🔴", "minor": "🟡", "ok": "🟢"}
GROUPS = {
    "Framing": ("crop line", "eye line", "headroom", "nose room"),
    "The head": ("head tilt", "camera height"),
    "Technical": ("eye focus", "face exposure", "light direction", "subject vs background"),
    "Behind them": (
        "subject separation", "background blur", "bright distractions",
        "competing faces", "colour distraction", "eye escape",
        "background clutter", "vertical intrusion", "horizontal intrusion",
    ),
}

st.title("⚖️ Shooti v4.2 — the model adjusts the judging")
st.caption(
    "The LLM now picks the style (which shifts every tolerance) and can dismiss or "
    "escalate individual findings, with guardrails and a full audit trail."
)

with st.sidebar:
    st.header("Photo")
    source = st.radio("Source", ["Camera", "Upload"], horizontal=True)
    frame_file = (
        st.camera_input("Frame your shot")
        if source == "Camera"
        else st.file_uploader("Portrait", type=["jpg", "jpeg", "png", "webp"])
    )

    st.header("AI coaching")
    st.caption("Optional. Everything else on this page is measured without an API.")

    # A real switch rather than a dropdown: the provider choice is a mode, and
    # the key/model fields below change meaning with it.
    provider = st.segmented_control(
        "Provider",
        options=list(PROVIDER_LABELS),
        format_func=lambda p: PROVIDER_LABELS[p],
        default="claude",
        key="provider",
    ) or "claude"

    api_key = st.text_input(
        f"{PROVIDER_LABELS[provider]} API key",
        type="password",
        key=f"key_{provider}",  # keys are kept separate per provider
        help=f"Or set {KEY_ENV[provider]} in the environment. Never committed anywhere.",
    )
    model = st.text_input(
        "Model",
        value=DEFAULT_MODELS[provider],
        key=f"model_{provider}",
        help="Change this if your account uses a different model name.",
    )
    intent = st.text_input(
        "Anything the model should know?",
        placeholder="editorial headshot, she wants it to feel approachable",
    )

    review = st.checkbox(
        "Let the model adjust the judging", True,
        help=f"Picks the style and may dismiss up to {MAX_DISMISS} findings it can see "
        f"are wrong. Without a key it falls back to keyword style selection and "
        "reviews nothing.",
    )
    style_override = st.selectbox(
        "Force a style",
        ["(let it choose)"] + [s.label for s in STYLES.values()],
    )
    ask = st.button("Get coaching", width="stretch", type="primary")
    if st.button("List models this key can call", width="stretch"):
        try:
            available = list_models(provider, api_key or None)
        except CoachError as exc:
            st.error(str(exc))
        else:
            st.success(f"{len(available)} models available")
            st.code("\n".join(available), language="text")

    st.header("Overlay")
    layers = {
        "show_landmarks": st.checkbox("Face landmarks (the evidence)", True),
        "show_body": st.checkbox("Predicted body landmarks + crop line", True),
        "show_eye_line": st.checkbox("Eye line vs upper third", True),
        "show_pose": st.checkbox("Head pose", True),
        "show_background": st.checkbox("Background: halo ring, hotspots, lines", True),
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

forced = None
if style_override != "(let it choose)":
    forced = next(s for s in STYLES.values() if s.label == style_override)

# Only recompute when something that actually affects the result changes.
# Previously every rerun re-ran the model review, so toggling an overlay
# checkbox fired a fresh LLM call and the page showed the PREVIOUS photo's
# result while it ran — which looks exactly like a stale/refresh bug.
photo_id = hashlib.sha1(frame_file.getvalue()).hexdigest()[:12]
signature = (
    photo_id,
    id(face) if len(faces) == 1 else faces.index(face),
    forced.key if forced else None,
    bool(review), provider, model, intent or "",
    hashlib.sha1((api_key or "").encode()).hexdigest()[:8],  # never store the key itself
)

if st.button("Re-analyse this photo", help="Force a fresh run, including a new model review."):
    st.session_state.pop("v42_sig", None)

if st.session_state.get("v42_sig") != signature:
    if forced is not None:
        with st.spinner("Measuring…"):
            verdict, adj = analyze_portrait(
                bgr, face, all_faces=faces, deep_background=True, style=forced
            ), None
    else:
        with st.spinner("Measuring and reviewing…"):
            raw_verdict = analyze_portrait(
                bgr, face, all_faces=faces, deep_background=True
            )
            verdict, adj = adjudicate(
                bgr, face, faces, raw_verdict,
                use_llm=review, provider=provider,
                api_key=api_key or None, model=model or None, intent=intent or None,
            )
    st.session_state["v42_sig"] = signature
    st.session_state["v42_result"] = (verdict, adj)
    st.session_state["v42_photo"] = getattr(frame_file, "name", photo_id)

verdict, adj = st.session_state["v42_result"]

# Provenance, so a stale result can never hide.
st.caption(
    f"Showing analysis of **{st.session_state.get('v42_photo', '?')}** "
    f"(`{photo_id}`). Overlay toggles redraw without re-running the model."
)

if adj is not None:
    a, b, c = st.columns(3)
    a.metric("Rules as written", f"{adj.raw_score}/100", help="Neutral tolerances, no review.")
    b.metric(
        f"Re-judged as {adj.style.label}", f"{adj.style_score}/100",
        delta=adj.style_delta or None,
        help="Same measurements, tolerances shifted to suit the style.",
    )
    c.metric(
        "After review", f"{adj.final_score}/100", delta=adj.review_delta or None,
        help=f"Findings the model dismissed or escalated. Dismissals capped at "
        f"+{MAX_SCORE_GAIN} points.",
    )
    st.caption(
        f"Style chosen by **{'the model' if adj.source == 'llm' else 'keyword fallback'}** — "
        f"{adj.style_reason}"
    )
    if adj.note:
        st.warning(adj.note)

    if adj.decisions:
        with st.expander(f"Review decisions ({len(adj.applied)} applied, "
                         f"{len(adj.blocked)} blocked)", expanded=True):
            for d in adj.decisions:
                if d.applied:
                    st.markdown(
                        f"{'🚫' if d.action == 'dismiss' else '⬆️'} **{d.action}** "
                        f"`{d.rule}` — {d.reason}"
                    )
                else:
                    st.markdown(
                        f"⛔ **blocked** `{d.rule}` ({d.action}) — guardrail: "
                        f"_{d.blocked_because}_. Model said: {d.reason}"
                    )
    elif adj.source == "llm":
        st.success("The model reviewed every finding and dismissed none.")
elif forced is not None:
    st.info(f"Style forced to **{forced.label}** — no model review ran.")

if verdict.human.verdict == "not-human":
    st.error(f"**Probably not a real person.** {verdict.human.note}")
elif verdict.human.verdict == "uncertain":
    st.warning(verdict.human.note)

left, right = st.columns([3, 2], gap="large")

with left:
    st.image(
        to_rgb(render(bgr, face, verdict.pose, verdict.crop,
                      background=verdict.background, **layers)),
        width="stretch",
    )
    st.caption(
        "Amber dots are the detected landmarks — if they are not on a face, ignore "
        "everything else. Green dashed lines are predicted body landmarks; red ones "
        "are joints you should not crop through. The magenta ellipse is the halo ring "
        "where separation is measured; red circles are blown patches."
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

    b = verdict.background
    if b is not None:
        st.markdown("**Background at a glance**")
        st.markdown(
            f"- separation: tone {b.separation_luma * 100:.0f}%, colour "
            f"{b.separation_color * 100:.0f}%{'  ⚠️ merging' if b.merges else ''}\n"
            f"- face is {b.blur_ratio:.2f}× sharper than the background\n"
            f"- clutter {b.clutter:.2f} · {len(b.hotspots)} blown patch(es) · "
            f"{b.competing_faces} rival face(s)"
        )

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

st.divider()

if ask:
    with st.spinner(f"Asking {PROVIDER_LABELS[provider]}…"):
        try:
            result = coach(
                bgr, verdict,
                provider=provider,
                api_key=api_key or None,
                model=model or None,
                intent=intent or None,
            )
        except CoachError as exc:
            st.error(str(exc))
        else:
            st.subheader(f"{PROVIDER_LABELS[result.provider]} says")
            st.markdown(result.text)
            tokens = (
                f" · {result.input_tokens} in / {result.output_tokens} out tokens"
                if result.input_tokens is not None
                else ""
            )
            st.caption(f"{result.model}{tokens}")
            st.caption(
                "The measurements above were sent as text alongside the photo, with "
                "instructions not to re-estimate them — so the model ranks and "
                "interprets rather than guessing at geometry."
            )
else:
    st.caption(
        f"Pick a provider in the sidebar, paste a key, and press **Get coaching** to "
        f"have {PROVIDER_LABELS[provider]} read the photo alongside these measurements."
    )

with st.expander("What gets sent to the model"):
    st.code(measurements_text(verdict, intent or None), language="text")

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
