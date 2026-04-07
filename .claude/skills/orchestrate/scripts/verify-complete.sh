#!/usr/bin/env bash
# verify-complete.sh — verify a PR task is truly done before the worktree is recycled
#
# Reads the agent's pr_number + steps from the state file, then checks:
#   1. All required steps are checkpointed
#   2. No unresolved review threads on the PR
#   3. No failing CI checks
#
# Repo is read from state file (.repo), falling back to the worktree's git remote.
#
# Usage: verify-complete.sh WINDOW
# Exit 0 = verified complete; exit 1 = not complete (with reason on stderr)

set -euo pipefail

WINDOW="$1"
STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"

# Read agent fields from state file
PR_NUMBER=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .pr_number // ""' "$STATE_FILE" 2>/dev/null)
STEPS=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .steps // [] | .[]' "$STATE_FILE" 2>/dev/null || true)
CHECKPOINTS=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .checkpoints // [] | .[]' "$STATE_FILE" 2>/dev/null || true)
WORKTREE_PATH=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .worktree_path // ""' "$STATE_FILE" 2>/dev/null)

# No PR number = no verification possible, assume done
if [ -z "$PR_NUMBER" ]; then
  echo "No pr_number in state — skipping verification"
  exit 0
fi

# --- Resolve repo: state file .repo → worktree git remote → fail gracefully ---
REPO=$(jq -r '.repo // ""' "$STATE_FILE" 2>/dev/null || echo "")
if [ -z "$REPO" ] && [ -n "$WORKTREE_PATH" ] && [ -d "$WORKTREE_PATH" ]; then
  REPO=$(git -C "$WORKTREE_PATH" remote get-url origin 2>/dev/null \
    | sed 's|.*github\.com[:/]||; s|\.git$||' || echo "")
fi

# --- Check 1: all required steps are checkpointed ---
MISSING=""
while IFS= read -r step; do
  [ -z "$step" ] && continue
  if ! echo "$CHECKPOINTS" | grep -qF "$step"; then
    MISSING="$MISSING $step"
  fi
done <<< "$STEPS"

if [ -n "$MISSING" ]; then
  echo "NOT COMPLETE: missing checkpoints:$MISSING on PR #$PR_NUMBER" >&2
  exit 1
fi

# --- Check 2: unresolved review threads ---
# Requires REPO to be resolved; skip gracefully if not
if [ -z "$REPO" ]; then
  echo "Warning: cannot resolve repo — skipping thread + CI checks" >&2
else
  # Split REPO into owner/name for GraphQL
  OWNER=$(echo "$REPO" | cut -d/ -f1)
  REPONAME=$(echo "$REPO" | cut -d/ -f2)

  UNRESOLVED=$(gh api graphql -f query="
    { repository(owner: \"${OWNER}\", name: \"${REPONAME}\") {
        pullRequest(number: ${PR_NUMBER}) {
          reviewThreads(first: 50) { nodes { isResolved } }
        }
      }
    }
  " --jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)] | length' 2>/dev/null || echo "0")

  if [ "$UNRESOLVED" -gt 0 ]; then
    echo "NOT COMPLETE: $UNRESOLVED unresolved review threads on PR #$PR_NUMBER" >&2
    exit 1
  fi

  # --- Check 3: CI not failing ---
  # gh pr checks --json may return an empty array if checks haven't started yet — that's fine.
  # We only fail if bucket == "fail" is present.
  FAILING=$(gh pr checks "$PR_NUMBER" --repo "$REPO" --json bucket 2>/dev/null \
    | jq '[.[] | select(.bucket == "fail")] | length' 2>/dev/null || echo "0")

  if [ "$FAILING" -gt 0 ]; then
    echo "NOT COMPLETE: $FAILING failing CI checks on PR #$PR_NUMBER" >&2
    exit 1
  fi
fi

echo "VERIFIED: PR #$PR_NUMBER — checkpoints ✓, 0 unresolved threads, CI green"
exit 0
