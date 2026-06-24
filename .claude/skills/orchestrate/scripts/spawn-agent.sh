#!/usr/bin/env bash
# spawn-agent.sh — create tmux window, checkout branch, launch claude, send task
#
# Usage: spawn-agent.sh SESSION WORKTREE_PATH SPARE_BRANCH NEW_BRANCH OBJECTIVE [PR_NUMBER] [STEPS...]
#   SESSION       — tmux session name, e.g. autogpt1
#   WORKTREE_PATH — absolute path to the git worktree
#   SPARE_BRANCH  — spare branch being replaced, e.g. spare/6 (saved for recycle)
#   NEW_BRANCH    — task branch to create, e.g. feat/my-feature
#   OBJECTIVE     — task description sent to the agent
#   PR_NUMBER     — (optional) GitHub PR number for completion verification
#   STEPS...      — (optional) required checkpoint names, e.g. pr-address pr-test
#
# Stdout: SESSION:WINDOW_INDEX (nothing else — callers rely on this)
# Exit non-zero on failure.

set -euo pipefail

if [ $# -lt 5 ]; then
  echo "Usage: spawn-agent.sh SESSION WORKTREE_PATH SPARE_BRANCH NEW_BRANCH OBJECTIVE [PR_NUMBER] [STEPS...]" >&2
  exit 1
fi

SESSION="$1"
WORKTREE_PATH="$2"
SPARE_BRANCH="$3"
NEW_BRANCH="$4"
OBJECTIVE="$5"
PR_NUMBER="${6:-}"
STEPS=("${@:7}")
WORKTREE_NAME=$(basename "$WORKTREE_PATH")
STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"

# Generate a stable session ID so this agent's Claude session can always be resumed:
#   claude --resume $SESSION_ID --permission-mode bypassPermissions
SESSION_ID=$(uuidgen 2>/dev/null || python3 -c "import uuid; print(uuid.uuid4())")

# Create (or switch to) the task branch
git -C "$WORKTREE_PATH" checkout -b "$NEW_BRANCH" 2>/dev/null \
  || git -C "$WORKTREE_PATH" checkout "$NEW_BRANCH"

# Open a new named tmux window; capture its numeric index
WIN_IDX=$(tmux new-window -t "$SESSION" -n "$WORKTREE_NAME" -P -F '#{window_index}')
WINDOW="${SESSION}:${WIN_IDX}"

# Append the initial agent record to the state file so subsequent jq updates find it.
# This must happen before the pr_number/steps update below.
if [ -f "$STATE_FILE" ]; then
  NOW=$(date +%s)
  jq --arg window "$WINDOW" \
     --arg worktree "$WORKTREE_NAME" \
     --arg worktree_path "$WORKTREE_PATH" \
     --arg spare_branch "$SPARE_BRANCH" \
     --arg branch "$NEW_BRANCH" \
     --arg objective "$OBJECTIVE" \
     --arg session_id "$SESSION_ID" \
     --argjson now "$NOW" \
     '.agents += [{
       "window": $window,
       "worktree": $worktree,
       "worktree_path": $worktree_path,
       "spare_branch": $spare_branch,
       "branch": $branch,
       "objective": $objective,
       "session_id": $session_id,
       "state": "running",
       "checkpoints": [],
       "last_output_hash": "",
       "last_seen_at": $now,
       "spawned_at": $now,
       "idle_since": 0,
       "revision_count": 0,
       "last_rebriefed_at": 0
     }]' "$STATE_FILE" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "$STATE_FILE"
fi

# Store pr_number + steps in state file if provided (enables verify-complete.sh).
# The agent record was appended above so the jq select now finds it.
if [ -n "$PR_NUMBER" ] && [ -f "$STATE_FILE" ]; then
  if [ "${#STEPS[@]}" -gt 0 ]; then
    STEPS_JSON=$(printf '%s\n' "${STEPS[@]}" | jq -R . | jq -s .)
  else
    STEPS_JSON='[]'
  fi
  jq --arg w "$WINDOW" --arg pr "$PR_NUMBER" --argjson steps "$STEPS_JSON" \
    '.agents |= map(if .window == $w then . + {pr_number: $pr, steps: $steps, checkpoints: []} else . end)' \
    "$STATE_FILE" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "$STATE_FILE"
fi

# Launch claude with a stable session ID so it can always be resumed after a crash:
#   claude --resume SESSION_ID --permission-mode bypassPermissions
tmux send-keys -t "$WINDOW" "cd '${WORKTREE_PATH}' && claude --permission-mode bypassPermissions --session-id '${SESSION_ID}'" Enter

# wait_for_claude_idle — poll until the pane shows idle ❯ with no spinner in the last 3 lines.
# Returns 0 when idle, 1 on timeout.
_wait_idle() {
  local window="$1" timeout="${2:-60}" elapsed=0
  while (( elapsed < timeout )); do
    local cmd pane_tail
    cmd=$(tmux display-message -t "$window" -p '#{pane_current_command}' 2>/dev/null || echo "")
    pane=$(tmux capture-pane -t "$window" -p 2>/dev/null || echo "")
    pane_tail=$(echo "$pane" | tail -3)
    # Check full pane (not just tail) — 'Enter to confirm' dialog can appear above the last 3 lines
    if echo "$pane" | grep -q "Enter to confirm"; then
      tmux send-keys -t "$window" Down Enter
      sleep 2; (( elapsed += 2 )); continue
    fi
    if [[ "$cmd" == "node" ]] && \
       echo "$pane_tail" | grep -q "❯" && \
       ! echo "$pane_tail" | grep -qE '[✳✽✢✶·✻✼✿❋✤]|Running…|Compacting'; then
      return 0
    fi
    sleep 2; (( elapsed += 2 ))
  done
  return 1
}

# Wait up to 60s for claude to be fully interactive and idle (❯ visible, no spinner).
if ! _wait_idle "$WINDOW" 60; then
  echo "[spawn-agent] WARNING: timed out waiting for idle ❯ prompt on $WINDOW — sending objective anyway" >&2
fi

# Send the task. Split text and Enter — if combined, Enter can fire before the string
# is fully buffered, leaving the message stuck as "[Pasted text +N lines]" unsent.
tmux send-keys -t "$WINDOW" "${OBJECTIVE} Output each completed step as CHECKPOINT:<step-name>. When ALL steps are done, output ORCHESTRATOR:DONE on its own line."
sleep 0.3
tmux send-keys -t "$WINDOW" Enter

# Only output the window address — nothing else (callers parse this)
echo "$WINDOW"
