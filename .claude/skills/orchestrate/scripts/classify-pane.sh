#!/usr/bin/env bash
# classify-pane.sh — Classify the current state of a tmux pane
#
# Usage: classify-pane.sh <tmux-target>
#   tmux-target: e.g. "work:0", "work:1.0"
#
# Output (stdout): JSON object:
#   { "state": "running|idle|waiting_approval|complete", "reason": "...", "pane_cmd": "..." }
#
# Exit codes: 0=ok, 1=error (invalid target or tmux window not found)

set -euo pipefail

TARGET="${1:-}"

if [ -z "$TARGET" ]; then
  echo '{"state":"error","reason":"no target provided","pane_cmd":""}'
  exit 1
fi

# Validate tmux target format: session:window or session:window.pane
if ! [[ "$TARGET" =~ ^[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+(\.[0-9]+)?$ ]]; then
  echo '{"state":"error","reason":"invalid tmux target format","pane_cmd":""}'
  exit 1
fi

# Check session exists (use %%:* to extract session name from session:window)
if ! tmux list-windows -t "${TARGET%%:*}" &>/dev/null 2>&1; then
  echo '{"state":"error","reason":"tmux target not found","pane_cmd":""}'
  exit 1
fi

# Get the current foreground command in the pane
PANE_CMD=$(tmux display-message -t "$TARGET" -p '#{pane_current_command}' 2>/dev/null || echo "unknown")

# Capture and strip ANSI codes (use perl for cross-platform compatibility — BSD sed lacks \x1b support)
RAW=$(tmux capture-pane -t "$TARGET" -p -S -50 2>/dev/null || echo "")
CLEAN=$(echo "$RAW" | perl -pe 's/\x1b\[[0-9;]*[a-zA-Z]//g; s/\x1b\(B//g; s/\x1b\[\?[0-9]*[hl]//g; s/\r//g' \
  | grep -v '^[[:space:]]*$' || true)

# --- Check: explicit completion marker ---
# Must be on its own line (not buried in the objective text sent at spawn time).
if echo "$CLEAN" | grep -qE "^[[:space:]]*ORCHESTRATOR:DONE[[:space:]]*$"; then
  jq -n --arg cmd "$PANE_CMD" '{"state":"complete","reason":"ORCHESTRATOR:DONE marker found","pane_cmd":$cmd}'
  exit 0
fi

# --- Check: Claude Code approval prompt patterns ---
LAST_40=$(echo "$CLEAN" | tail -40)
APPROVAL_PATTERNS=(
  "Do you want to proceed"
  "Do you want to make this"
  "\\[y/n\\]"
  "\\[Y/n\\]"
  "\\[n/Y\\]"
  "Proceed\\?"
  "Allow this command"
  "Run bash command"
  "Allow bash"
  "Would you like"
  "Press enter to continue"
  "Esc to cancel"
)
for pattern in "${APPROVAL_PATTERNS[@]}"; do
  if echo "$LAST_40" | grep -qiE "$pattern"; then
    jq -n --arg pattern "$pattern" --arg cmd "$PANE_CMD" \
      '{"state":"waiting_approval","reason":"approval pattern: \($pattern)","pane_cmd":$cmd}'
    exit 0
  fi
done

# --- Check: shell prompt (claude has exited) ---
# If the foreground process is a shell (not claude/node), the agent has exited
case "$PANE_CMD" in
  zsh|bash|fish|sh|dash|tcsh|ksh)
    jq -n --arg cmd "$PANE_CMD" \
      '{"state":"idle","reason":"agent exited — shell prompt active","pane_cmd":$cmd}'
    exit 0
    ;;
esac

# Agent is still running (claude/node/python is the foreground process)
jq -n --arg cmd "$PANE_CMD" \
  '{"state":"running","reason":"foreground process: \($cmd)","pane_cmd":$cmd}'
exit 0
