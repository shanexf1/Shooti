#!/usr/bin/env bash
# Recreate the Shooti GitHub board: labels, milestones, issues.
#
# Idempotent — re-running skips anything that already exists, so it doubles as
# the version-controlled source of truth for the project plan.
#
# Usage:  ./scripts/gh_bootstrap.sh [repo]
#         repo defaults to the current directory's origin remote.

set -euo pipefail

GH="${GH:-$(command -v gh || echo "$HOME/.local/bin/gh")}"
if [ ! -x "$GH" ]; then
  echo "gh CLI not found. Install it or set GH=/path/to/gh" >&2
  exit 1
fi

REPO="${1:-}"
REPO_ARGS=()
[ -n "$REPO" ] && REPO_ARGS=(--repo "$REPO")

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- labels
say "Labels"
add_label() {
  if "$GH" label create "$1" --color "$2" --description "$3" "${REPO_ARGS[@]}" 2>/dev/null; then
    echo "  + $1"
  else
    "$GH" label edit "$1" --color "$2" --description "$3" "${REPO_ARGS[@]}" >/dev/null 2>&1 \
      && echo "  = $1" || echo "  ! $1 (skipped)"
  fi
}

add_label "area:cv"    "1d76db" "Computer vision, detection, geometry"
add_label "area:llm"   "8250df" "Claude prompting and coaching output"
add_label "area:ui"    "0e8a16" "Streamlit interface and overlays"
add_label "area:eval"  "fbca04" "Test sets, calibration, accuracy"
add_label "area:infra" "5319e7" "Tooling, packaging, CI"
add_label "area:docs"  "c5def5" "README, write-up, demo"
add_label "p0"         "b60205" "Must land for a working submission"
add_label "p1"         "d93f0b" "Strongly wanted"
add_label "p2"         "e99695" "Stretch"

# ------------------------------------------------------------ milestones
say "Milestones"
MILESTONE_PATH="repos/{owner}/{repo}/milestones"
[ -n "$REPO" ] && MILESTONE_PATH="repos/$REPO/milestones"

add_milestone() {
  # POST returns 422 when the milestone already exists, which is the skip case.
  if "$GH" api "$MILESTONE_PATH" -f title="$1" -f description="$2" >/dev/null 2>&1; then
    echo "  + $1"
  else
    echo "  = $1 (exists)"
  fi
}

add_milestone "M1 Harden the MVP" \
  "The analyze-a-photo path works end to end. Make it trustworthy: calibrate thresholds, test the direction conventions, fix subject extent."
add_milestone "M2 Real-time guidance" \
  "Move from analyzing a captured frame to coaching a live preview. This is the feature that makes the project more than after-the-fact feedback."
add_milestone "M3 Coaching quality" \
  "More composition patterns than thirds, and evidence that the Claude prompt actually helps."
add_milestone "M4 Submission" \
  "Write-up, demo, and a setup path verified on a clean machine."

# ---------------------------------------------------------------- issues
say "Issues"
existing="$("$GH" issue list --state all --limit 200 --json title --jq '.[].title' "${REPO_ARGS[@]}" 2>/dev/null || true)"

new_issue() {
  local title="$1" milestone="$2" labels="$3" body="$4"
  if printf '%s\n' "$existing" | grep -Fxq "$title"; then
    echo "  = $title"
    return
  fi
  "$GH" issue create \
    --title "$title" \
    --body "$body" \
    --milestone "$milestone" \
    --label "$labels" \
    "${REPO_ARGS[@]}" >/dev/null
  echo "  + $title"
}

# ---- M1 -------------------------------------------------------------------
new_issue "Unit-test the pan/tilt/roll direction conventions" \
  "M1 Harden the MVP" "area:cv,p0" \
'The move directions invert twice and are easy to get backwards. One inversion
(raising the camera pushes the subject DOWN in frame) was already shipped wrong
once and caught by hand.

Add tests that pin all four:
- subject left of target -> pan camera LEFT
- subject right of target -> pan camera RIGHT
- subject below target -> LOWER the camera
- subject above target -> RAISE the camera
- horizon rising to the right -> roll COUNTER-CLOCKWISE

Build them from synthetic frames with a known subject position so the expected
answer is unambiguous.'

new_issue "Build a 25-photo labeled test set" \
  "M1 Harden the MVP" "area:eval,p0" \
'Shooti has no ground truth, so every threshold is currently an assertion.

Collect ~25 photos spanning: portraits (tight/loose), full-body, groups,
landscapes with a clear horizon, indoor shots with no horizon, and deliberately
bad framing. For each, record by hand: where the subject is, whether the horizon
is level, and whether the framing is good/bad and why.

Store as `eval/labels.json` plus the images (or URLs, if licensing is awkward).'

new_issue "Calibrate rule thresholds against the test set" \
  "M1 Harden the MVP" "area:eval,p1" \
'Depends on the labeled test set.

THIRDS_TOL, TILT_TOL_DEG, HEADROOM_IDEAL, PITCH_TOL, SUBJECT_SIZE_RANGE and the
balance skew cutoff are all hand-set from photographic convention. Measure
false-positive and false-negative rates per rule and tune. Report before/after
numbers in the write-up — a rule that fires on good photos is worse than no
rule.'

new_issue "Fix subject-extent underestimation on full-body shots" \
  "M1 Harden the MVP" "area:cv,p1" \
'With a face detected, subject extent is the face-implied head-and-shoulders box
unioned with an overlapping saliency blob. On a full-body action shot this still
under-covers the person, so `subject size` advises "move closer" when the subject
already fills the frame.

Options: a person-segmentation model (adds a dependency), or grow the box along
the dominant energy axis from the face downward. Evaluate both against the test
set rather than picking by feel.'

new_issue "Handle group photos properly" \
  "M1 Harden the MVP" "area:cv,p2" \
'With multiple faces, the subject box is the union of all faces and the anchor is
the largest face. That means a group is composed around whoever stands closest to
the camera.

Better: anchor on the centroid of the face cluster, and check that the group as a
whole is centered/balanced rather than applying the single-subject thirds rule.'

# ---- M2 -------------------------------------------------------------------
new_issue "Live camera loop with continuous guidance" \
  "M2 Real-time guidance" "area:ui,p0" \
'The headline claim is guidance *while* shooting, but the app currently analyzes
a captured frame.

Add a live mode (streamlit-webrtc, or a small FastAPI + getUserMedia page) that
runs the CV pipeline on every Nth frame and draws the overlay on the preview.
Target 10 fps on a laptop; the rule engine is cheap, YuNet is the cost, so
consider running detection every 3rd frame and interpolating.'

new_issue "Throttle and cache the Claude call for live mode" \
  "M2 Real-time guidance" "area:llm,p0" \
'Calling Claude per frame would be both slow and expensive.

Only call when the framing has materially changed (anchor moved >5% of frame, or
score changed by >8), and rate-limit to at most one call every few seconds. Cache
the last response and keep showing it while the frame is stable. Show the token
count so the cost is visible during the demo.'

new_issue "Directional HUD with a lock-on state" \
  "M2 Real-time guidance" "area:ui,p1" \
'In live mode, arrows on a still image are hard to act on. Draw a persistent HUD:
edge-anchored arrows for the pan/tilt direction that shrink as you approach the
target, and a clear "locked" state when every rule is inside tolerance so the
user knows when to press the shutter.'

# ---- M3 -------------------------------------------------------------------
new_issue "Add leading lines, symmetry, and diagonal detection" \
  "M3 Coaching quality" "area:cv,p1" \
'Thirds is one pattern among many, and applying it to a symmetrical shot gives
actively bad advice.

The Hough transform already runs for the horizon — reuse it to find converging
lines (leading lines toward the subject) and strong verticals. Add a symmetry
check by comparing left/right halves, and suppress the thirds penalty when the
frame is clearly a symmetric composition.'

new_issue "Offer phi-grid targets as an alternative to thirds" \
  "M3 Coaching quality" "area:cv,p2" \
'Some photographers compose on the golden ratio (0.382/0.618) rather than
0.333/0.667. Make the target grid selectable and show both, then note in the
write-up whether the choice changes the advice enough to matter.'

new_issue "A/B the Claude prompt against the test set" \
  "M3 Coaching quality" "area:llm,p1" \
'Claim to verify: giving Claude the CV measurements produces better advice than
handing it the bare image.

Run three variants over the test set — image only, measurements only, and both
(current) — and score the outputs on whether the moves are correct and
actionable. This is the strongest evidence available for the design bet in the
README, and it is cheap to run.'

# ---- M4 -------------------------------------------------------------------
new_issue "Write up strengths, weaknesses, and next steps" \
  "M4 Submission" "area:docs,p0" \
'Required by the assignment. Cover: what the split between measured geometry and
LLM judgment bought, where it breaks (saliency picking the wrong subject, no
pitch estimate indoors), the calibration numbers from M1, and what a v2 would do
differently.

Include the direction-inversion bug as a concrete example of why the geometry
layer needs tests.'

new_issue "Record a demo video and capture screenshots" \
  "M4 Submission" "area:docs,p0" \
'Short screen recording: a badly framed shot, the overlay explaining why, the
corrected shot with a higher score. Add stills to the README so it reads well
without running anything.'

new_issue "Verify setup on a clean machine" \
  "M4 Submission" "area:infra,p1" \
'Fresh venv, no cached model, no API key. Confirm: deps install, the ONNX model
downloads, the smoke test passes, the app boots, and the no-API-key path is
genuinely usable. Fix whatever the README leaves out.'

say "Done. Board: $("$GH" repo view --json url --jq .url "${REPO_ARGS[@]}" 2>/dev/null)/issues"
