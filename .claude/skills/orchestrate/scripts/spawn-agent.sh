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
     --argjson now "$NOW" \
     '.agents += [{
       "window": $window,
       "worktree": $worktree,
       "worktree_path": $worktree_path,
       "spare_branch": $spare_branch,
       "branch": $branch,
       "objective": $objective,
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

# Build recovery instruction — agent reads state file + gh pr view to reorient after compaction
RECOVERY=""
if [ -n "$PR_NUMBER" ]; then
  RECOVERY=" If your context compacts and you lose track of what to do, run: cat ~/.claude/orchestrator-state.json | jq '.agents[] | select(.window==\"$WINDOW\")' and gh pr view $PR_NUMBER --json title,body,headRefName to reorient. Output each completed step as a checkpoint: CHECKPOINT:<step-name> on its own line."
fi

# Launch claude — single-quote path so spaces and special chars are safe
tmux send-keys -t "$WINDOW" "cd '${WORKTREE_PATH}' && claude --permission-mode bypassPermissions" Enter

# Wait up to 60s for claude to be fully interactive:
# both pane_current_command == 'node' AND the '❯' prompt is visible.
PROMPT_FOUND=false
for i in $(seq 1 60); do
  CMD=$(tmux display-message -t "$WINDOW" -p '#{pane_current_command}' 2>/dev/null || echo "")
  PANE=$(tmux capture-pane -t "$WINDOW" -p 2>/dev/null || echo "")
  if echo "$PANE" | grep -q "Enter to confirm"; then
    tmux send-keys -t "$WINDOW" Down Enter
    sleep 2
    continue
  fi
  if [[ "$CMD" == "node" ]] && echo "$PANE" | grep -q "❯"; then
    PROMPT_FOUND=true
    break
  fi
  sleep 1
done

if ! $PROMPT_FOUND; then
  echo "[spawn-agent] WARNING: timed out waiting for ❯ prompt on $WINDOW — sending objective anyway" >&2
fi

# Send the task with checkpoint protocol and recovery instructions.
# Split text and Enter into separate calls — if sent together, Enter can fire before
# the full string is buffered into Claude's input, leaving the message stuck unsent.
tmux send-keys -t "$WINDOW" "${OBJECTIVE}${RECOVERY} When ALL steps are done, output ORCHESTRATOR:DONE on its own line."
sleep 0.3
tmux send-keys -t "$WINDOW" Enter

# Only output the window address — nothing else (callers parse this)
echo "$WINDOW"
