#!/usr/bin/env bash
# Publish Shooti to GitHub: create the repo, push main, then build the board.
#
# Prerequisite (interactive, one time):
#   gh auth login        # GitHub.com -> HTTPS -> login with a web browser
#
# Idempotent: safe to re-run. Skips repo creation if it already exists, and
# gh_bootstrap.sh skips labels/milestones/issues that already exist.

set -euo pipefail

REPO_NAME="${REPO_NAME:-Shooti}"
VISIBILITY="${VISIBILITY:---public}"
DESCRIPTION="${DESCRIPTION:-AI photography assistant that measures your framing and tells you how to move the camera}"

GH="${GH:-$(command -v gh || echo "$HOME/.local/bin/gh")}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

if [ ! -x "$GH" ]; then
  echo "gh CLI not found. Set GH=/path/to/gh" >&2
  exit 1
fi

if ! "$GH" auth status >/dev/null 2>&1; then
  cat >&2 <<EOF
Not logged in to GitHub. Run this first, then re-run this script:

    $GH auth login

Choose: GitHub.com -> HTTPS -> Yes (authenticate git) -> login with a web browser
EOF
  exit 1
fi

OWNER="$("$GH" api user --jq .login)"
SLUG="$OWNER/$REPO_NAME"

say "Repo: $SLUG"
if "$GH" repo view "$SLUG" >/dev/null 2>&1; then
  echo "  = already exists"
  git remote get-url origin >/dev/null 2>&1 \
    || git remote add origin "https://github.com/$SLUG.git"
  git push -u origin main
else
  # --source/--push wires up the remote and pushes the current branch in one go.
  "$GH" repo create "$REPO_NAME" \
    "$VISIBILITY" \
    --description "$DESCRIPTION" \
    --source=. \
    --remote=origin \
    --push
  echo "  + created and pushed"
fi

say "Board"
"$ROOT/scripts/gh_bootstrap.sh" "$SLUG"

say "Live at https://github.com/$SLUG"
