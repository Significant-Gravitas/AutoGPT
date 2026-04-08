#!/usr/bin/env bash
# recycle-agent.sh — kill a tmux window and restore the worktree to its spare branch
#
# Usage: recycle-agent.sh WINDOW WORKTREE_PATH SPARE_BRANCH
#   WINDOW        — tmux target, e.g. autogpt1:3
#   WORKTREE_PATH — absolute path to the git worktree
#   SPARE_BRANCH  — branch to restore, e.g. spare/6
#
# Stdout: one status line

set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: recycle-agent.sh WINDOW WORKTREE_PATH SPARE_BRANCH" >&2
  exit 1
fi

WINDOW="$1"
WORKTREE_PATH="$2"
SPARE_BRANCH="$3"

# Kill the tmux window (ignore error — may already be gone)
tmux kill-window -t "$WINDOW" 2>/dev/null || true

# Restore to spare branch: abort any in-progress operation, then clean
git -C "$WORKTREE_PATH" rebase --abort 2>/dev/null || true
git -C "$WORKTREE_PATH" merge --abort 2>/dev/null || true
git -C "$WORKTREE_PATH" reset --hard HEAD 2>/dev/null
git -C "$WORKTREE_PATH" clean -fd 2>/dev/null
git -C "$WORKTREE_PATH" checkout "$SPARE_BRANCH"

echo "Recycled: $(basename "$WORKTREE_PATH") → $SPARE_BRANCH (window $WINDOW closed)"
