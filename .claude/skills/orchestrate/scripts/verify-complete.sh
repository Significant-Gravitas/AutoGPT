#!/usr/bin/env bash
# verify-complete.sh — verify a PR task is truly done before marking the agent done
#
# Check order matters:
#   1. Checkpoints — did the agent do all required steps?
#   2. CI complete — no pending (bots post comments AFTER their check runs, must wait)
#   3. CI passing — no failures (agent must fix before done)
#   4. spawned_at — a new CI run was triggered after agent spawned (proves real work)
#   5. Unresolved threads — checked AFTER CI so bot-posted comments are included
#   6. CHANGES_REQUESTED — checked AFTER CI so bot reviews are included
#
# Usage: verify-complete.sh WINDOW
# Exit 0 = verified complete; exit 1 = not complete (stderr has reason)

set -euo pipefail

WINDOW="$1"
STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"

PR_NUMBER=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .pr_number // ""' "$STATE_FILE" 2>/dev/null)
STEPS=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .steps // [] | .[]' "$STATE_FILE" 2>/dev/null || true)
CHECKPOINTS=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .checkpoints // [] | .[]' "$STATE_FILE" 2>/dev/null || true)
WORKTREE_PATH=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .worktree_path // ""' "$STATE_FILE" 2>/dev/null)
BRANCH=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .branch // ""' "$STATE_FILE" 2>/dev/null)
SPAWNED_AT=$(jq -r --arg w "$WINDOW" '.agents[] | select(.window == $w) | .spawned_at // "0"' "$STATE_FILE" 2>/dev/null || echo "0")

# No PR number = cannot verify
if [ -z "$PR_NUMBER" ]; then
  echo "NOT COMPLETE: no pr_number in state — set pr_number or mark done manually" >&2
  exit 1
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

# Resolve repo for all GitHub checks below
REPO=$(jq -r '.repo // ""' "$STATE_FILE" 2>/dev/null || echo "")
if [ -z "$REPO" ] && [ -n "$WORKTREE_PATH" ] && [ -d "$WORKTREE_PATH" ]; then
  REPO=$(git -C "$WORKTREE_PATH" remote get-url origin 2>/dev/null \
    | sed 's|.*github\.com[:/]||; s|\.git$||' || echo "")
fi

if [ -z "$REPO" ]; then
  echo "Warning: cannot resolve repo — skipping CI/thread checks" >&2
  echo "VERIFIED: PR #$PR_NUMBER — checkpoints ✓ (CI/thread checks skipped — no repo)"
  exit 0
fi

CI_BUCKETS=$(gh pr checks "$PR_NUMBER" --repo "$REPO" --json bucket 2>/dev/null || echo "[]")

# --- Check 2: CI fully complete — no pending checks ---
# Pending checks MUST finish before we check threads/reviews:
# bots (Seer, Check PR Status, etc.) post comments and CHANGES_REQUESTED AFTER their CI check runs.
PENDING=$(echo "$CI_BUCKETS" | jq '[.[] | select(.bucket == "pending")] | length' 2>/dev/null || echo "0")
if [ "$PENDING" -gt 0 ]; then
  PENDING_NAMES=$(gh pr checks "$PR_NUMBER" --repo "$REPO" --json bucket,name 2>/dev/null \
    | jq -r '[.[] | select(.bucket == "pending") | .name] | join(", ")' 2>/dev/null || echo "unknown")
  echo "NOT COMPLETE: $PENDING CI checks still pending on PR #$PR_NUMBER ($PENDING_NAMES)" >&2
  exit 1
fi

# --- Check 3: CI passing — no failures ---
FAILING=$(echo "$CI_BUCKETS" | jq '[.[] | select(.bucket == "fail")] | length' 2>/dev/null || echo "0")
if [ "$FAILING" -gt 0 ]; then
  FAILING_NAMES=$(gh pr checks "$PR_NUMBER" --repo "$REPO" --json bucket,name 2>/dev/null \
    | jq -r '[.[] | select(.bucket == "fail") | .name] | join(", ")' 2>/dev/null || echo "unknown")
  echo "NOT COMPLETE: $FAILING failing CI checks on PR #$PR_NUMBER ($FAILING_NAMES)" >&2
  exit 1
fi

# --- Check 4: a new CI run was triggered AFTER the agent spawned ---
if [ -n "$BRANCH" ] && [ "${SPAWNED_AT:-0}" -gt 0 ]; then
  LATEST_RUN_AT=$(gh run list --repo "$REPO" --branch "$BRANCH" \
    --json createdAt --limit 1 2>/dev/null | jq -r '.[0].createdAt // ""')
  if [ -n "$LATEST_RUN_AT" ]; then
    if date --version >/dev/null 2>&1; then
      LATEST_RUN_EPOCH=$(date -d "$LATEST_RUN_AT" "+%s" 2>/dev/null || echo "0")
    else
      LATEST_RUN_EPOCH=$(TZ=UTC date -j -f "%Y-%m-%dT%H:%M:%SZ" "$LATEST_RUN_AT" "+%s" 2>/dev/null || echo "0")
    fi
    if [ "$LATEST_RUN_EPOCH" -le "$SPAWNED_AT" ]; then
      echo "NOT COMPLETE: latest CI run on $BRANCH predates agent spawn — agent may not have pushed yet" >&2
      exit 1
    fi
  fi
fi

OWNER=$(echo "$REPO" | cut -d/ -f1)
REPONAME=$(echo "$REPO" | cut -d/ -f2)

# --- Check 5: no unresolved review threads (checked AFTER CI — bots post after their check) ---
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

# --- Check 6: no CHANGES_REQUESTED (checked AFTER CI — bots post reviews after their check) ---
# A CHANGES_REQUESTED review is stale if the latest commit was pushed AFTER the review was submitted.
# Stale reviews (pre-dating the fixing commits) should not block verification.
#
# Fetch commits and latestReviews in a single call and fail closed — if gh fails,
# treat that as NOT COMPLETE rather than silently passing.
# Use latestReviews (not reviews) so each reviewer's latest state is used — superseded
# CHANGES_REQUESTED entries are automatically excluded when the reviewer later approved.
# Note: we intentionally use committedDate (not PR updatedAt) because updatedAt changes on any
# PR activity (bot comments, label changes) which would create false negatives.
PR_REVIEW_METADATA=$(gh pr view "$PR_NUMBER" --repo "$REPO" \
  --json commits,latestReviews 2>/dev/null) || {
  echo "NOT COMPLETE: unable to fetch PR review metadata for PR #$PR_NUMBER" >&2
  exit 1
}

LATEST_COMMIT_DATE=$(jq -r '.commits[-1].committedDate // ""' <<< "$PR_REVIEW_METADATA")
CHANGES_REQUESTED_REVIEWS=$(jq '[.latestReviews[]? | select(.state == "CHANGES_REQUESTED")]' <<< "$PR_REVIEW_METADATA")

BLOCKING_CHANGES_REQUESTED=0
BLOCKING_REQUESTERS=""

if [ -n "$LATEST_COMMIT_DATE" ] && [ "$(echo "$CHANGES_REQUESTED_REVIEWS" | jq length)" -gt 0 ]; then
  if date --version >/dev/null 2>&1; then
    LATEST_COMMIT_EPOCH=$(date -d "$LATEST_COMMIT_DATE" "+%s" 2>/dev/null || echo "0")
  else
    LATEST_COMMIT_EPOCH=$(TZ=UTC date -j -f "%Y-%m-%dT%H:%M:%SZ" "$LATEST_COMMIT_DATE" "+%s" 2>/dev/null || echo "0")
  fi

  while IFS= read -r review; do
    [ -z "$review" ] && continue
    REVIEW_DATE=$(echo "$review" | jq -r '.submittedAt // ""')
    REVIEWER=$(echo "$review" | jq -r '.author.login // "unknown"')
    if [ -z "$REVIEW_DATE" ]; then
      # No submission date — treat as fresh (conservative: blocks verification)
      BLOCKING_CHANGES_REQUESTED=$(( BLOCKING_CHANGES_REQUESTED + 1 ))
      BLOCKING_REQUESTERS="${BLOCKING_REQUESTERS:+$BLOCKING_REQUESTERS, }${REVIEWER}"
    else
      if date --version >/dev/null 2>&1; then
        REVIEW_EPOCH=$(date -d "$REVIEW_DATE" "+%s" 2>/dev/null || echo "0")
      else
        REVIEW_EPOCH=$(TZ=UTC date -j -f "%Y-%m-%dT%H:%M:%SZ" "$REVIEW_DATE" "+%s" 2>/dev/null || echo "0")
      fi
      if [ "$REVIEW_EPOCH" -gt "$LATEST_COMMIT_EPOCH" ]; then
        # Review was submitted AFTER latest commit — still fresh, blocks verification
        BLOCKING_CHANGES_REQUESTED=$(( BLOCKING_CHANGES_REQUESTED + 1 ))
        BLOCKING_REQUESTERS="${BLOCKING_REQUESTERS:+$BLOCKING_REQUESTERS, }${REVIEWER}"
      fi
      # Review submitted BEFORE latest commit — stale, skip
    fi
  done <<< "$(echo "$CHANGES_REQUESTED_REVIEWS" | jq -c '.[]')"
else
  # No commit date or no changes_requested — check raw count as fallback
  BLOCKING_CHANGES_REQUESTED=$(echo "$CHANGES_REQUESTED_REVIEWS" | jq length 2>/dev/null || echo "0")
  BLOCKING_REQUESTERS=$(echo "$CHANGES_REQUESTED_REVIEWS" | jq -r '[.[].author.login] | join(", ")' 2>/dev/null || echo "unknown")
fi

if [ "$BLOCKING_CHANGES_REQUESTED" -gt 0 ]; then
  echo "NOT COMPLETE: CHANGES_REQUESTED (after latest commit) from ${BLOCKING_REQUESTERS} on PR #$PR_NUMBER" >&2
  exit 1
fi

echo "VERIFIED: PR #$PR_NUMBER — checkpoints ✓, CI complete + green, 0 unresolved threads, no CHANGES_REQUESTED"
exit 0
