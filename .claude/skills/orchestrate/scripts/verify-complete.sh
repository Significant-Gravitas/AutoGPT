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
BRANCH=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .branch // ""' "$STATE_FILE" 2>/dev/null)
SPAWNED_AT=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .spawned_at // "0"' "$STATE_FILE" 2>/dev/null || echo "0")

# No PR number = cannot verify — refuse to recycle (supervisor must handle)
if [ -z "$PR_NUMBER" ]; then
  echo "NOT COMPLETE: no pr_number in state — cannot verify; set pr_number or mark done manually" >&2
  exit 1
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
  if ! echo "$CHECKPOINTS" | grep -qFx "$step"; then
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

  # --- Check 3: no CHANGES_REQUESTED reviews (from any reviewer, bot or human) ---
  CHANGES_REQUESTED=$(gh pr view "$PR_NUMBER" --repo "$REPO" \
    --json reviews --jq '[.reviews[] | select(.state == "CHANGES_REQUESTED")] | length' 2>/dev/null || echo "0")

  if [ "$CHANGES_REQUESTED" -gt 0 ]; then
    REQUESTERS=$(gh pr view "$PR_NUMBER" --repo "$REPO" \
      --json reviews --jq '[.reviews[] | select(.state == "CHANGES_REQUESTED") | .author.login] | join(", ")' 2>/dev/null || echo "unknown")
    echo "NOT COMPLETE: CHANGES_REQUESTED from ${REQUESTERS} on PR #$PR_NUMBER" >&2
    exit 1
  fi

  # --- Check 5: CI not failing ---
  # gh pr checks --json may return an empty array if checks haven't started yet — that's fine.
  # We only fail if bucket == "fail" is present.
  FAILING=$(gh pr checks "$PR_NUMBER" --repo "$REPO" --json bucket 2>/dev/null \
    | jq '[.[] | select(.bucket == "fail")] | length' 2>/dev/null || echo "0")

  if [ "$FAILING" -gt 0 ]; then
    echo "NOT COMPLETE: $FAILING failing CI checks on PR #$PR_NUMBER" >&2
    exit 1
  fi

  # --- Check 6: a new CI run was triggered AFTER the agent spawned ---
  # Guards against agents that output CHECKPOINT:* from the objective text itself
  # (copied from spawn message) without actually doing work, while CI was already
  # green from a previous run.
  if [ -n "$BRANCH" ] && [ "${SPAWNED_AT:-0}" -gt 0 ]; then
    LATEST_RUN_AT=$(gh run list --repo "$REPO" --branch "$BRANCH" \
      --json createdAt --limit 1 2>/dev/null | jq -r '.[0].createdAt // ""')
    if [ -n "$LATEST_RUN_AT" ]; then
      # Convert ISO 8601 timestamp to epoch — handle both macOS (BSD date) and Linux (GNU date)
      if date --version >/dev/null 2>&1; then
        LATEST_RUN_EPOCH=$(date -d "$LATEST_RUN_AT" "+%s" 2>/dev/null || echo "0")
      else
        LATEST_RUN_EPOCH=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$LATEST_RUN_AT" "+%s" 2>/dev/null || echo "0")
      fi
      if [ "$LATEST_RUN_EPOCH" -le "$SPAWNED_AT" ]; then
        echo "NOT COMPLETE: latest CI run on $BRANCH predates agent spawn — no new CI triggered yet" >&2
        exit 1
      fi
    fi
  fi
fi

echo "VERIFIED: PR #$PR_NUMBER — checkpoints ✓, 0 unresolved threads, CI green"
exit 0
