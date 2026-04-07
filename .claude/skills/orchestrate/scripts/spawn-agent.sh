#!/usr/bin/env bash
# spawn-agent.sh — create tmux window, checkout branch, launch claude, send task
#
# Usage: spawn-agent.sh SESSION WORKTREE_PATH SPARE_BRANCH NEW_BRANCH OBJECTIVE
#   SESSION       — tmux session name, e.g. autogpt1
#   WORKTREE_PATH — absolute path to the git worktree
#   SPARE_BRANCH  — spare branch being replaced, e.g. spare/6 (saved for recycle)
#   NEW_BRANCH    — task branch to create, e.g. feat/my-feature
#   OBJECTIVE     — task description sent to the agent
#
# Stdout: SESSION:WINDOW_INDEX (nothing else — callers rely on this)
# Exit non-zero on failure.

set -euo pipefail

if [ $# -lt 5 ]; then
  echo "Usage: spawn-agent.sh SESSION WORKTREE_PATH SPARE_BRANCH NEW_BRANCH OBJECTIVE" >&2
  exit 1
fi

SESSION="$1"
WORKTREE_PATH="$2"
SPARE_BRANCH="$3"
NEW_BRANCH="$4"
OBJECTIVE="$5"
WORKTREE_NAME=$(basename "$WORKTREE_PATH")

# Create (or switch to) the task branch
git -C "$WORKTREE_PATH" checkout -b "$NEW_BRANCH" 2>/dev/null \
  || git -C "$WORKTREE_PATH" checkout "$NEW_BRANCH"

# Open a new named tmux window; capture its numeric index
WIN_IDX=$(tmux new-window -t "$SESSION" -n "$WORKTREE_NAME" -P -F '#{window_index}')
WINDOW="${SESSION}:${WIN_IDX}"

# Launch claude — single-quote path so spaces and special chars are safe
tmux send-keys -t "$WINDOW" "cd '${WORKTREE_PATH}' && claude --permission-mode bypassPermissions" Enter

# Wait up to 60s for claude to be fully interactive:
# both pane_current_command == 'node' AND the '❯' prompt is visible.
# Checking the prompt avoids sending the objective to the shell before
# claude finishes initializing (which silently drops the message).
for i in $(seq 1 60); do
  CMD=$(tmux display-message -t "$WINDOW" -p '#{pane_current_command}' 2>/dev/null || echo "")
  PANE=$(tmux capture-pane -t "$WINDOW" -p 2>/dev/null || echo "")
  # Auto-dismiss settings error dialog if present
  if echo "$PANE" | grep -q "Enter to confirm"; then
    tmux send-keys -t "$WINDOW" Down Enter
    sleep 2
    continue
  fi
  # Ready when node is running AND the interactive prompt is visible
  if [[ "$CMD" == "node" ]] && echo "$PANE" | grep -q "❯"; then
    break
  fi
  sleep 1
done

# Send the task — agent must output ORCHESTRATOR:DONE when finished
tmux send-keys -t "$WINDOW" "${OBJECTIVE}. When all work is done, output the exact string: ORCHESTRATOR:DONE" Enter

# Only output the window address — nothing else (callers parse this)
echo "$WINDOW"
