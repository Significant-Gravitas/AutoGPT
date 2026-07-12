#!/usr/bin/env bash
# notify.sh — send a fleet notification message
#
# Delivery order (first available wins):
#   1. Discord webhook — DISCORD_WEBHOOK_URL env var OR state file .discord_webhook
#   2. macOS notification center — osascript (silent fail if unavailable)
#   3. Stdout only
#
# Usage: notify.sh MESSAGE
# Exit: always 0 (notification failure must not abort the caller)

MESSAGE="${1:-}"
[ -z "$MESSAGE" ] && exit 0

STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"

# --- Resolve Discord webhook ---
WEBHOOK="${DISCORD_WEBHOOK_URL:-}"
if [ -z "$WEBHOOK" ] && [ -f "$STATE_FILE" ]; then
  WEBHOOK=$(jq -r '.discord_webhook // ""' "$STATE_FILE" 2>/dev/null || echo "")
fi

# --- Discord delivery ---
if [ -n "$WEBHOOK" ]; then
  PAYLOAD=$(jq -n --arg msg "$MESSAGE" '{"content": $msg}')
  curl -s -X POST "$WEBHOOK" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" > /dev/null 2>&1 || true
fi

# --- macOS notification center (silent if not macOS or osascript missing) ---
if command -v osascript &>/dev/null 2>&1; then
  # Escape single quotes for AppleScript
  SAFE_MSG=$(echo "$MESSAGE" | sed "s/'/\\\\'/g")
  osascript -e "display notification \"${SAFE_MSG}\" with title \"Orchestrator\"" 2>/dev/null || true
fi

# Always print to stdout so run-loop.sh logs it
echo "$MESSAGE"
exit 0
