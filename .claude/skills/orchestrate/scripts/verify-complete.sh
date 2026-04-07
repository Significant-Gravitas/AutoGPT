#!/usr/bin/env bash
# verify-complete.sh — verify a PR task is truly done before the worktree is recycled
#
# Reads the agent's pr_number + steps from the state file, then checks:
#   1. All required steps are checkpointed
#   2. No unresolved review threads on the PR
#   3. No failing CI checks
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

# No PR number = no verification possible, assume done
if [ -z "$PR_NUMBER" ]; then
  echo "No pr_number in state — skipping verification" >&2
  exit 0
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
UNRESOLVED=$(gh api graphql -f query="
  { repository(owner: \"Significant-Gravitas\", name: \"AutoGPT\") {
      pullRequest(number: $PR_NUMBER) {
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
FAILING=$(gh pr checks "$PR_NUMBER" --repo Significant-Gravitas/AutoGPT --json bucket 2>/dev/null \
  | jq '[.[] | select(.bucket == "fail")] | length' 2>/dev/null || echo "0")

if [ "$FAILING" -gt 0 ]; then
  echo "NOT COMPLETE: $FAILING failing CI checks on PR #$PR_NUMBER" >&2
  exit 1
fi

echo "VERIFIED: PR #$PR_NUMBER — checkpoints ✓, 0 unresolved threads, CI green"
exit 0
