"""Shooti v2 — learned grading instead of rule violations.

v1 asserted that an off-thirds subject was a defect. On 5,110 human-rated photos
that assumption did not hold (thirds distance vs. rating: SRCC -0.02, p=0.08),
and v1's whole rule score correlated with human judgment at SRCC -0.006.

v2 replaces the judge: a model trained on human ratings grades the photo, and
advice comes from actually trying reframings and keeping the ones that score
higher. If nothing scores higher, it says so instead of inventing a violation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from shooti import analyze
from shooti.grader.advise import candidates, suggest
from shooti.grader.grade import (
    CHANNEL_LABELS,
    CKPT_DIR,
    available_channels,
    grade_all_channels,
    load_grader,
)
from shooti.loader import load_bgr
from shooti.overlay import to_rgb

st.set_page_config(page_title="Shooti v2", page_icon="🎯", layout="wide")


@st.cache_data(show_spinner=False)
def load_report() -> dict:
    for name in ("report_test.json", "report.json"):
        path = CKPT_DIR / name
        if path.exists():
            return {"name": name, **json.loads(path.read_text())}
    return {}


st.title("🎯 Shooti v2")
st.caption(
    "Graded by a model trained on 20,437 human-rated photos — not by hand-written rules."
)

report = load_report()

with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["Camera", "Upload"], horizontal=True)
    frame_file = (
        st.camera_input("Frame your shot")
        if source == "Camera"
        else st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"])
    )

    st.header("Model")
    # Label these by what they actually mean: which INPUT CHANNELS feed the head,
    # not how many graders run. "both" read as "both graders" and confused people.
    channels = st.selectbox(
        "Which grader drives the advice",
        available_channels(),
        format_func=lambda c: CHANNEL_LABELS.get(c, c),
        help="Which inputs the grading head sees. CLIP + geometry is the most "
        "responsive to reframing, so it gives the best advice. CLIP only grades "
        "marginally better. Geometry only is a weak baseline, kept for comparison. "
        "All three score your photo below regardless of this choice.",
    )
    top_k = st.slider("Max suggestions", 1, 5, 3)
    min_gain = st.slider(
        "Minimum gain to suggest", 0.0, 0.5, 0.04, 0.01,
        help="How much better a reframing must score before it's worth mentioning. "
        "Raise it to only hear about big wins.",
    )
    show_v1 = st.checkbox("Show v1 rule verdict for comparison", True)

try:
    load_grader(channels)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if report:
    learned = report.get("learned", {}).get(channels, {})
    base = report.get("baseline_rules", {})
    cols = st.columns(4)
    cols[0].metric(
        "Model agreement (SRCC)",
        f"{learned.get('srcc', float('nan')):.3f}",
        help=f"Rank correlation with human mean rating on {report.get('n', '?')} "
        f"held-out photos ({report['name']}).",
    )
    cols[1].metric(
        "v1 rules (SRCC)",
        f"{base.get('srcc', float('nan')):.3f}",
        help="The hand-written rule score on the same photos. Near zero.",
    )
    cols[2].metric("Good/bad accuracy", f"{learned.get('acc', 0) * 100:.1f}%")
    premise = report.get("premise_check", {})
    cols[3].metric(
        "Thirds → quality",
        f"{premise.get('thirds_dist_srcc', float('nan')):+.3f}",
        help=f"Correlation between distance-from-thirds and human rating "
        f"(p={premise.get('thirds_dist_p', float('nan')):.2g}). "
        "This is v1's core assumption, and it does not hold.",
    )

if frame_file is None:
    st.info("Take a photo or upload one to get started.")
    st.stop()

bgr = load_bgr(frame_file.getvalue())

with st.spinner(f"Grading and searching {len(candidates())} reframings…"):
    base_grade, suggestions = suggest(
        bgr, channels=channels, top_k=top_k, min_gain=min_gain
    )

left, right = st.columns([3, 2], gap="large")

with left:
    st.image(to_rgb(bgr), width="stretch", caption="As shot")

with right:
    st.metric("Predicted human rating", f"{base_grade.score:.2f} / 10")
    st.progress(min(1.0, max(0.0, (base_grade.score - 1) / 9)))

    st.caption(
        f"Vote spread {base_grade.spread:.2f}"
        + (" — divisive, people would disagree about this one." if base_grade.divisive else "")
    )

    # Bin index 0 is rating 1, so pass explicit x values — otherwise the axis
    # reads 0-9 and mislabels every bar.
    st.bar_chart(
        pd.DataFrame(
            {
                "rating": np.arange(1, len(base_grade.distribution) + 1),
                "share of voters": base_grade.distribution,
            }
        ),
        x="rating",
        y="share of voters",
    )

st.divider()

st.subheader("What each grader says about this photo")
st.caption(
    "Same photo, three heads trained on different inputs. They are scored in one "
    "pass, so this costs nothing extra. Where they disagree is informative: the "
    "geometry-only head can only see framing, so a big gap means the image content "
    "is doing the work, not the composition."
)

all_grades = grade_all_channels(bgr)
gcols = st.columns(len(all_grades) + 1)
for col, (name, g) in zip(gcols, all_grades.items()):
    learned_srcc = report.get("learned", {}).get(name, {}).get("srcc")
    col.metric(
        CHANNEL_LABELS.get(name, name) + (" ← driving advice" if name == channels else ""),
        f"{g.score:.2f} / 10",
        help=f"Held-out agreement with human ratings: SRCC "
        f"{learned_srcc:.3f}." if learned_srcc is not None else None,
    )
    col.caption(f"vote spread {g.spread:.2f}")

# The v1 rule score belongs in this row too — it is the thing being compared.
v1_analysis = analyze(bgr)
gcols[-1].metric(
    "v1 rules (no signal)",
    f"{v1_analysis.score} / 100",
    help="Shown on a 0-100 scale because that is what v1 emits. It correlates "
    "with human judgment at SRCC -0.006, so treat it as a curiosity.",
)
gcols[-1].caption("not comparable — different scale")

st.divider()

if not suggestions:
    st.success(
        f"**No reframing scored higher.** The model tried {len(candidates()) - 1} "
        "alternatives — pans, tighter crops, and rolls — and none beat your framing "
        f"by at least {min_gain:.2f}. This is where v1 would still have complained "
        "about the subject not sitting on a thirds line."
    )
else:
    n = len(suggestions)
    st.subheader(
        f"{n} reframing that scores higher" if n == 1 else f"{n} reframings that score higher"
    )
    st.caption(
        "Each was actually cropped and re-graded. The gain is the predicted rating "
        "difference, not a rule penalty."
    )
    # Fixed column count so a single suggestion doesn't render full-width.
    cols = st.columns(max(3, len(suggestions)))
    for col, sug in zip(cols, suggestions):
        with col:
            st.image(
                to_rgb(sug.crop_bgr),
                width="stretch",
                caption=f"{sug.candidate.label} → {sug.grade.score:.2f} ({sug.gain:+.2f})",
            )
            st.markdown(f"**{sug.candidate.advice}**")

    st.info(
        "**What this can't tell you:** only crops *inside* your existing frame can be "
        "tested, because pixels outside it were never captured. So 'step back' or "
        "'zoom out' is never suggested — not because it wouldn't help, but because "
        "it can't be measured from one photo."
    )

if show_v1:
    st.divider()
    with st.expander("v1 rule verdict on this photo (for comparison)"):
        v1 = v1_analysis  # already computed for the comparison row above
        st.markdown(f"v1 rule score: **{v1.score}/100**")
        problems = v1.problems
        if problems:
            for f in problems:
                st.markdown(f"- **{f.rule}** — {f.message}  \n  → {f.action}")
        else:
            st.markdown("_v1 found nothing to fix._")
        st.caption(
            "Kept only to show the contrast. These findings are asserted from "
            "hand-set thresholds and, measured against human ratings, carry no "
            "predictive signal (SRCC -0.006)."
        )
