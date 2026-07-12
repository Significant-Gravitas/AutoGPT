#!/usr/bin/env bash
# status.sh — print orchestrator status: state file summary + live tmux pane commands
#
# Usage: status.sh
# Reads: ~/.claude/orchestrator-state.json

set -euo pipefail

STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"

if [ ! -f "$STATE_FILE" ] || ! jq -e '.' "$STATE_FILE" >/dev/null 2>&1; then
  echo "No orchestrator state found at $STATE_FILE"
  exit 0
fi

# Header: active status, session, thresholds, last poll
jq -r '
  "=== Orchestrator [\(if .active then "RUNNING" else "STOPPED" end)] ===",
  "Session: \(.tmux_session // "unknown")  |  Idle threshold: \(.idle_threshold_seconds // 300)s",
  "Last poll: \(if (.last_poll_at // 0) == 0 then "never" else (.last_poll_at | strftime("%H:%M:%S")) end)",
  ""
' "$STATE_FILE"

# Each agent: state, window, worktree/branch, truncated objective
AGENT_COUNT=$(jq '.agents | length' "$STATE_FILE")
if [ "$AGENT_COUNT" -eq 0 ]; then
  echo "  (no agents registered)"
else
  jq -r '
    .agents[] |
    "  [\(.state | ascii_upcase)] \(.window)  \(.worktree)/\(.branch)",
    "    \(.objective // "" | .[0:70])"
  ' "$STATE_FILE"
fi

echo ""

# Live pane_current_command for non-done agents
while IFS= read -r WINDOW; do
  [ -z "$WINDOW" ] && continue
  CMD=$(tmux display-message -t "$WINDOW" -p '#{pane_current_command}' 2>/dev/null || echo "unreachable")
  echo "  $WINDOW live: $CMD"
done < <(jq -r '.agents[] | select(.state != "done") | .window' "$STATE_FILE" 2>/dev/null || true)
