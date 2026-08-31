"""Shooti v3 — you say what you're going for, and the rules follow.

The lineage:
  v1  one rule set for every photo. Its score correlates with human ratings at
      SRCC -0.006. It scolded a symmetric photo for being centered.
  v2  a grader learned from 20,437 human-rated photos. SRCC 0.682. Good scores,
      but the advice is a crop search rather than something explainable.
  v3  the photographer states their intent; Claude picks the matching rule set.
      Advice becomes intent-appropriate -- a symmetry shot is now rewarded for
      centering, and a street photograph is not scolded for a cant.

What v3 does NOT claim: that the rule score predicts quality. Measured with
CLIP-inferred intent it still does not (SRCC +0.007, p=0.62). So the QUALITY
number here still comes from the v2 learned grader, and the rule sets supply
interpretable, intent-appropriate feedback instead of a verdict.
"""

from __future__ import annotations

import json

import streamlit as st

from shooti import analyze
from shooti.grader.grade import CHANNEL_LABELS, CKPT_DIR, available_channels, grade_all_channels
from shooti.grader.features import features_from_analysis
from shooti.intent import PROFILES, evaluate, select
from shooti.intent.select import select_keyword
from shooti.loader import load_bgr
from shooti.overlay import to_rgb

st.set_page_config(page_title="Shooti v3", page_icon="🎬", layout="wide")

SEVERITY_ICON = {"major": "🔴", "minor": "🟡", "ok": "🟢"}
SOURCE_LABEL = {
    "claude": "Claude read your intent and the photo",
    "keyword": "keyword fallback (no LLM call)",
    "default": "no intent given",
    "manual": "you chose it",
}


@st.cache_data(show_spinner=False)
def load_reports() -> tuple[dict, dict]:
    grader = CKPT_DIR / "report_test.json"
    intent = CKPT_DIR / "report_intent_validation.json"
    return (
        json.loads(grader.read_text()) if grader.exists() else {},
        json.loads(intent.read_text()) if intent.exists() else {},
    )


grader_report, intent_report = load_reports()

st.title("🎬 Shooti v3")
st.caption("Tell it what you're going for, and it applies the rules that fit — not one universal set.")

with st.sidebar:
    st.header("Your shot")
    source = st.radio("Source", ["Camera", "Upload"], horizontal=True)
    frame_file = (
        st.camera_input("Frame your shot")
        if source == "Camera"
        else st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"])
    )
    intent = st.text_area(
        "What are you going for?",
        placeholder="a symmetric shot of the cathedral hallway\n"
        "a moody portrait of my friend at dusk\n"
        "candid street photo, I want it to feel unposed",
        height=90,
    )

    st.header("Rule-set selection")
    use_llm = st.checkbox(
        "Ask Claude to pick the rule set", True,
        help="Needs an Anthropic API key. Without one it falls back to keyword "
        "matching automatically and tells you it did.",
    )
    api_key = st.text_input("Anthropic API key", type="password",
                            help="Or set ANTHROPIC_API_KEY. Leave blank to use the environment.")
    override = st.selectbox(
        "Override the rule set",
        ["(let it choose)"] + [f"{p.label}" for p in PROFILES.values()],
        help="Force a specific rule set, whatever the selector says.",
    )

    st.header("Grading")
    channels = st.selectbox(
        "Which learned grader scores the photo",
        available_channels(),
        format_func=lambda c: CHANNEL_LABELS.get(c, c),
    )

if frame_file is None:
    st.info("Add a photo and describe what you're going for.")
    if intent.strip():
        preview = select_keyword(intent)
        st.caption(
            f"On your description alone, the keyword fallback would pick "
            f"**{preview.profile.label}** — {preview.profile.blurb}"
        )
    st.stop()

bgr = load_bgr(frame_file.getvalue())

# One detection pass feeds both the v1 comparison and the v3 rule sets.
base_analysis = analyze(bgr)
features = features_from_analysis(base_analysis, bgr)
signed_tilt = base_analysis.horizon.angle_deg if base_analysis.horizon else None

if override != "(let it choose)":
    profile = next(p for p in PROFILES.values() if p.label == override)
    from shooti.intent.select import Selection

    selection = Selection(profile, "manual", "You picked this rule set directly.")
else:
    with st.spinner("Choosing the rule set for this shot…"):
        selection = select(bgr, intent, use_llm=use_llm, api_key=api_key or None)

verdict = evaluate(features, selection.profile, horizon_angle_signed=signed_tilt)

# "for a no stated intent" reads badly; the generic profile is not a noun phrase.
_label = selection.profile.label.lower()
_article = "an" if _label[0] in "aeiou" else "a"
intent_phrase = "this shot" if selection.profile.key == "generic" else f"{_article} {_label}"

# ------------------------------------------------------------------ the header
c1, c2 = st.columns([2, 3])
with c1:
    st.metric("Rule set applied", selection.profile.label)
    st.caption(f"Chosen by: {SOURCE_LABEL.get(selection.source, selection.source)}")
with c2:
    st.markdown(f"**Why:** {selection.reasoning}")
    if selection.runner_up:
        st.caption(f"Runner-up considered: {selection.runner_up.label}")
    if selection.note:
        st.warning(selection.note)

st.caption(f"_{selection.profile.blurb}_")

left, right = st.columns([3, 2], gap="large")

with left:
    st.image(to_rgb(bgr), width="stretch", caption="As shot")

with right:
    grades = grade_all_channels(bgr)
    g = grades.get(channels)
    if g:
        st.metric("Predicted human rating", f"{g.score:.2f} / 10")
        st.progress(min(1.0, max(0.0, (g.score - 1) / 9)))
        st.caption(
            f"From the v2 learned grader (SRCC "
            f"{grader_report.get('learned', {}).get(channels, {}).get('srcc', float('nan')):.3f}). "
            "This is the quality estimate — the rule set below shapes the *advice*, not this number."
        )

    st.metric(f"{selection.profile.label} rule score", f"{verdict.score} / 100")
    st.caption(
        "Interpretable, but do not read it as quality: intent-conditioned rule scores "
        "measured at SRCC "
        f"{intent_report.get('conditioned_srcc', float('nan')):+.3f} "
        f"(p={intent_report.get('conditioned_p', float('nan')):.2g}) against human ratings."
    )

st.divider()

problems = verdict.problems
if problems:
    st.subheader(f"{len(problems)} thing{'s' if len(problems) > 1 else ''} to change for {intent_phrase}")
    for f in problems:
        with st.container(border=True):
            st.markdown(f"{SEVERITY_ICON[f.severity]} **{f.rule}** — {f.message}")
            st.markdown(f"→ {f.action}")
else:
    st.success(f"Nothing to change — this frame satisfies every rule that applies to {intent_phrase}.")

if verdict.skipped:
    st.info(
        f"**Not applicable to this intent:** {', '.join(verdict.skipped)}. "
        "These rules are switched off, not scored leniently — that is the difference "
        "from v1, which applied all of them to every photograph."
    )

with st.expander("What the universal v1 rule set would have said instead"):
    st.markdown(f"v1 score: **{base_analysis.score}/100** (vs {verdict.score} under {selection.profile.label})")
    v1_problems = base_analysis.problems
    if v1_problems:
        for f in v1_problems:
            still = any(p.rule == f.rule for p in problems)
            mark = "also flagged" if still else "**dropped by this intent**"
            st.markdown(f"- **{f.rule}** — {f.message}  \n  _{mark}_")
    else:
        st.markdown("_v1 found nothing to fix._")

with st.expander("Compare all rule sets on this photo"):
    st.caption("Same measurements, every rule set. The spread is the point of v3.")
    rows = []
    for key, p in PROFILES.items():
        v = evaluate(features, p, horizon_angle_signed=signed_tilt)
        rows.append({
            "rule set": p.label,
            "score": v.score,
            "problems": len(v.problems),
            "rules off": len(v.skipped),
            "applied": "←" if key == selection.profile.key else "",
        })
    st.dataframe(rows, hide_index=True, width="stretch")
