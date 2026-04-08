#!/usr/bin/env bash
# capacity.sh — show fleet capacity: available spare worktrees + in-use agents
#
# Usage: capacity.sh [REPO_ROOT]
#   REPO_ROOT defaults to the root worktree of the current git repo.
#
# Reads: ~/.claude/orchestrator-state.json (skipped if missing or corrupt)

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"
REPO_ROOT="${1:-$(git rev-parse --show-toplevel 2>/dev/null || echo "")}"

echo "=== Available (spare) worktrees ==="
if [ -n "$REPO_ROOT" ]; then
  SPARE=$("$SCRIPTS_DIR/find-spare.sh" "$REPO_ROOT" 2>/dev/null || echo "")
else
  SPARE=$("$SCRIPTS_DIR/find-spare.sh" 2>/dev/null || echo "")
fi

if [ -z "$SPARE" ]; then
  echo "  (none)"
else
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    echo "  ✓ $line"
  done <<< "$SPARE"
fi

echo ""
echo "=== In-use worktrees ==="
if [ -f "$STATE_FILE" ] && jq -e '.' "$STATE_FILE" >/dev/null 2>&1; then
  IN_USE=$(jq -r '.agents[] | select(.state != "done") | "  [\(.state)] \(.worktree_path) → \(.branch)"' \
    "$STATE_FILE" 2>/dev/null || echo "")
  if [ -n "$IN_USE" ]; then
    echo "$IN_USE"
  else
    echo "  (none)"
  fi
else
  echo "  (no active state file)"
fi
