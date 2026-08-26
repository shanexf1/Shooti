# 📷 Shooti

An AI photography assistant that tells you how to *move* — not just what's wrong.

Point it at a photo (or shoot one in the browser) and it measures the framing,
then hands back concrete camera moves: pan left 11% of the frame, roll 6°
counter-clockwise, lower the camera, give the subject looking room.

Built for Project 0. It demonstrates the two required properties:

- **Intelligence** — an eight-rule composition engine over real computer vision:
  DNN face detection with facial landmarks, gradient-energy saliency for
  non-face subjects, Hough-transform horizon detection for roll *and* camera
  pitch, and a weighted 0-100 score. Every number is measured, not guessed.
- **Interaction** — a camera/upload GUI that draws the analysis back onto your
  photo (thirds grid, subject box, eye line, and an arrow from where the subject
  *is* to where it *should be*), plus optional Claude vision coaching that
  prioritizes the three moves worth making.

## The design bet

Most photo-feedback tools describe a photo. Shooti's split is deliberate:

**Geometry is measured, never asked of the LLM.** The CV layer computes position,
roll, pitch, headroom, subject size, and balance, and produces a score with no
API key required. Those measurements are then handed to Claude *as text
alongside the image*, with instructions not to re-estimate them. Claude does only
what geometry can't: recognize what the subject actually is, read the light and
background, and rank the fixes.

This means the app is fully useful offline, the advice is reproducible, and the
LLM can't hallucinate a pixel coordinate.

## Setup

Requires Python 3.10+ (3.13 recommended — the `anthropic` SDK dropped 3.9).

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

The face detector downloads a 227 KB ONNX model into `models/` on first run. If
that download fails, face detection degrades to the saliency path instead of
crashing.

Claude coaching is optional. To enable it, either export a key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

or paste one into the sidebar, or run `ant auth login` (the SDK picks up the
profile automatically). Without any of these, the score and rule findings still
work — only the "Ask Claude for moves" button needs credentials.

## Verify it without the UI

```bash
.venv/bin/python scripts/smoke_test.py
```

Runs the full CV pipeline over synthetic scenes plus anything you drop in
`samples/`, prints every measurement, and writes annotated overlays to `out/`.
No API key needed.

## What the rules check

| Rule | Measured from | Output |
|---|---|---|
| thirds | eye level (people) or subject centroid | pan/raise/lower, in % of frame |
| tilt | dominant near-horizontal Hough line | roll angle + direction to correct |
| camera pitch | horizon offset from center row | high/low angle detection |
| headroom | estimated head top vs. frame top | tilt or height change |
| looking room | nose offset from eye midpoint (head yaw) | pan to open space ahead of the gaze |
| edges | subject box vs. frame border | crop warnings |
| subject size | subject area / frame area | move closer or step back |
| balance | left/right gradient-energy split | pan or reposition subject |

**Direction convention** (fixed once, in `shooti/rules.py`): panning the camera
left moves the frame left, so the subject appears further right. Every move
string follows from that, including the vertical case — raising the camera pushes
the subject *down* in frame. This inversion is easy to get backwards and is
worth a unit test (see [issue tracker](../../issues)).

## Layout

```
app.py                 Streamlit UI
shooti/subject.py      YuNet face detection + landmarks, saliency fallback
shooti/rules.py        the eight rules, the score, the direction conventions
shooti/overlay.py      draws grid, subject box, eye line, target, move arrow
shooti/coach.py        Claude vision call (claude-opus-5) + measurement grounding
shooti/loader.py       EXIF-aware image decoding
scripts/smoke_test.py  runs the pipeline, no API key
scripts/gh_bootstrap.sh  creates the GitHub labels, milestones, and issues
```

## Known limits

Honest list, since the assignment asks for strengths *and* weaknesses:

- **Subject extent is approximate.** With a face detected, Shooti unions the
  face-implied head-and-shoulders box with an overlapping saliency blob to
  estimate the body. On a full-body action shot it still under-covers the
  subject, so "subject size" advice can read low. A segmentation model would fix
  this properly.
- **Saliency is not semantics.** With no face in frame, "the subject" is the
  highest-energy compact gradient blob. On a busy background it can pick the
  wrong thing. It reports `none` rather than guessing when the frame is flat.
- **Pitch detection needs a horizon.** The camera-angle estimate comes from where
  the horizon crosses frame center, so it silently abstains indoors.
- **Thresholds are hand-tuned, not fitted.** They were set from photographic
  convention and spot-checked, not calibrated against a labeled set. That
  calibration is tracked as open work.
- **Not yet real-time.** It analyzes a captured frame, not a live preview. The
  live-guidance loop is the main remaining feature.

## Project management

Work is tracked in GitHub Issues and milestones. `scripts/gh_bootstrap.sh`
recreates the whole board from scratch, so the plan is version-controlled
alongside the code.
