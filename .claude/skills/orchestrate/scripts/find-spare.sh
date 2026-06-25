#!/usr/bin/env bash
# find-spare.sh — list worktrees on spare/N branches (free to use)
#
# Usage: find-spare.sh [REPO_ROOT]
#   REPO_ROOT defaults to the root worktree containing the current git repo.
#
# Output (stdout): one line per available worktree: "PATH BRANCH"
#   e.g.: /Users/me/Code/AutoGPT3 spare/3

set -euo pipefail

REPO_ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || echo "")}"
if [ -z "$REPO_ROOT" ]; then
  echo "Error: not inside a git repo and no REPO_ROOT provided" >&2
  exit 1
fi

git -C "$REPO_ROOT" worktree list --porcelain \
  | awk '
      /^worktree / { path = substr($0, 10) }
      /^branch /   { branch = substr($0, 8); print path " " branch }
    ' \
  | { grep -E " refs/heads/spare/[0-9]+$" || true; } \
  | sed 's|refs/heads/||'
