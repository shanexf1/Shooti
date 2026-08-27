# 📷 Shooti

An AI photography assistant that tells you how to *move* — not just what's wrong.

> **v2 is here, and it disproved v1.** `app2.py` grades photos with a model
> trained on 20,437 human-rated images instead of hand-written rules. Measured
> on 5,110 held-out photos, v1's rule score correlated with human judgment at
> **SRCC −0.006** — no better than noise — while the learned grader reaches
> **0.682**. See [Results](#v2-results-what-the-data-said) below. v1 is kept
> runnable for comparison, not because it works.

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

---

# v2 — learned grading

v1's premise was that composition rules define quality: off-thirds is worse,
tilted is worse, and a weighted sum of violations is a score. v2 tests that
premise against human ratings, and it does not survive.

## v2 results: what the data said

Trained on the AVA subset (`trojblue/AVA-aesthetics-10pct-min50-10bins`, every
photo with 50+ human votes). Fit on 17,371 images, best epoch chosen on 3,066
carved out of train, and the numbers below measured on **5,110 photos used for
neither fitting nor selection**.

| Model | SRCC ↑ | PLCC ↑ | MSE ↓ | good/bad acc ↑ |
|---|---|---|---|---|
| **v1 hand-tuned rules** | **−0.006** | 0.008 | 5.014 | 65.9% |
| Learned: geometry only | 0.166 | 0.166 | 0.538 | 71.4% |
| Learned: CLIP only | **0.682** | 0.689 | 0.290 | 80.1% |
| Learned: CLIP + geometry | 0.672 | 0.680 | 0.298 | **80.6%** |

Three findings, including one that is bad news for the approach:

1. **v1's rule score carries no signal.** SRCC −0.006 across 5,110 photos. Not
   "too strict about thirds" — as a predictor of human judgment it was noise.
2. **Thirds distance does not predict quality.** SRCC −0.025, **p = 0.077**, so
   not significant even at the 5% level. Distance from dead center is likewise
   flat (−0.017, p = 0.23). The single assumption v1 was built on is unsupported.
3. **The geometric features add nothing on top of CLIP** (0.672 vs 0.682). The
   18 hand-engineered measurements are redundant once the model can see the
   image. That is a negative result about my own feature engineering, and it is
   reported rather than buried.

Calibration for the headline number: published NIMA-style models reach roughly
SRCC 0.6–0.7 on full AVA, so 0.682 is in the expected range. It is *not* directly
comparable to those papers — this subset keeps only photos with 50+ votes, which
makes the labels cleaner than the full set.

## How v2 gives advice without rules

The grader alone would be a black box that says "6.2/10" and nothing actionable.
So advice comes from **counterfactual search** instead of rule violations:

1. Build 14 candidate reframings — pans, tighter crops, rolls.
2. Actually crop each one and re-grade it with the model.
3. Report only the ones that score higher than the frame as shot.

This is what fixes the complaint that started v2. A centered or symmetric photo
that cannot be improved by shifting simply produces **no shift suggestion** — no
rule needs special-casing. In a 12-photo spot check, the model declined to
suggest anything for 3 of them; v1 flagged nearly all of them.

Two things were verified rather than assumed:

- **The model responds to reframing.** Mean spread of predicted scores across
  candidates was 0.51 (`both`) and 0.44 (`clip`), never flat. Had it been ~0, the
  suggestions would have been meaningless.
- **No synthetic pixels are ever graded.** Rotated candidates crop to 74% scale,
  verified to stay inside the frame, so the model never scores reflected or black
  borders.

`both` is the default for advice despite grading marginally worse, because it is
measurably more responsive to framing changes (0.51 vs 0.44) — geometry gives the
model an explicit handle on the thing the advice is about.

## Both versions on the same photo

![v2](docs/screenshot-v2.png)

v2 predicts a 5.75/10 human rating, shows the full predicted vote distribution
(so you can see how divisive a shot is, not just its average), and finds exactly
one reframing worth making — a 7% downward tilt, worth +0.07.

![v1](docs/screenshot-v1.png)

v1 on the identical photo asserts "2 things to fix": the eye level is 12% off a
thirds intersection, and there is 23% headroom. Neither claim is connected to
whether anyone likes the photograph.

## Running v2

```bash
.venv/bin/pip install -r requirements-v2.txt
.venv/bin/python -m streamlit run app2.py
```

Both screenshots above were produced by `scripts/drive_app.py`, which launches
the app in a real headless browser, uploads a photo through the actual file
input, and screenshots the result — so "it works" means the UI rendered, not
that a port opened:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/playwright install chromium
.venv/bin/python scripts/drive_app.py --app app2.py --photo samples/messi.jpg
```

The trained checkpoints are committed (1.3 MB), so this works immediately. To
reproduce training from scratch (~15 min, ~3.4 GB of transient download):

```bash
python -m shooti.grader.prepare --split train       # ~5 min at 70 img/s
python -m shooti.grader.prepare --split validation  # ~1 min
python -m shooti.grader.train --ablate              # seconds
```

Add `--cleanup` to `prepare` to delete each parquet shard after use, which keeps
peak disk around 0.5 GB instead of 3.4 GB.

## v2 limits

- **Only crops inside the existing frame can be tested.** "Step back" or "zoom
  out" is never suggested — not because it wouldn't help, but because the pixels
  outside your frame were never captured, so the model cannot score it.
- **AVA is DPChallenge, not your photos.** It skews toward contest-style images
  from the 2000s. A grader trained on it inherits that taste, and there is no
  reason to expect it to match any individual photographer's.
- **SRCC 0.68 is useful, not authoritative.** It ranks photos far better than
  chance and far better than v1, but it will disagree with you on individual
  shots. It is a second opinion, not a verdict.
- **Held-out ≠ a third split.** This dataset's `test` split is a byte-identical
  copy of its `validation` split (verified: all 5,110 ids match), so the
  selection set is carved out of train instead. Anyone extending this should not
  treat `test` as extra data.
- **`both` vs `clip` is a real tradeoff,** not a free win: better advice
  responsiveness, slightly worse grading.

## Layout (v2 additions)

```
app2.py                      Streamlit UI for the learned grader
shooti/grader/features.py    the same measurements as v1, as features not penalties
shooti/grader/embed.py       frozen CLIP ViT-B/32 image embeddings
shooti/grader/model.py       NIMA-style distribution head + EMD loss
shooti/grader/prepare.py     builds the feature cache from AVA parquet shards
shooti/grader/train.py       training, ablation, and the honest split discipline
shooti/grader/eval.py        re-score a checkpoint on any cached split
shooti/grader/grade.py       inference
shooti/grader/advise.py      counterfactual reframing search
checkpoints/report_test.json every number in the table above
```

## Project management

Work is tracked in GitHub Issues and milestones. `scripts/gh_bootstrap.sh`
recreates the whole board from scratch, so the plan is version-controlled
alongside the code.
