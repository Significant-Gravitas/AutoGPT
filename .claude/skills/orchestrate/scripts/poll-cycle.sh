#!/usr/bin/env bash
# poll-cycle.sh — Single orchestrator poll cycle
#
# Reads ~/.claude/orchestrator-state.json, classifies each agent, updates state,
# and outputs a JSON array of actions for Claude to take.
#
# Usage: poll-cycle.sh
# Output (stdout): JSON array of action objects
#   [{ "window": "work:0", "action": "kick|approve|none", "state": "...",
#      "worktree": "...", "objective": "...", "reason": "..." }]
#
# The state file is updated in-place (atomic write via .tmp).

set -uo pipefail

STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFY="$SCRIPTS_DIR/classify-pane.sh"

# Cross-platform md5
md5_hash() {
  if command -v md5sum &>/dev/null; then
    md5sum | awk '{print $1}'
  else
    md5
  fi
}

# Ensure state file exists
if [ ! -f "$STATE_FILE" ]; then
  echo '{"active":false,"agents":[]}' > "$STATE_FILE"
fi

ACTIVE=$(jq -r '.active // false' "$STATE_FILE")
if [ "$ACTIVE" != "true" ]; then
  echo "[]"
  exit 0
fi

NOW=$(date +%s)
IDLE_THRESHOLD=$(jq -r '.idle_threshold_seconds // 300' "$STATE_FILE")

ACTIONS="[]"
UPDATED_AGENTS="[]"

# Read agents as newline-delimited JSON objects
AGENTS_JSON=$(jq -c '.agents[]' "$STATE_FILE" 2>/dev/null || echo "")

if [ -z "$AGENTS_JSON" ]; then
  echo "[]"
  exit 0
fi

while IFS= read -r agent; do
  [ -z "$agent" ] && continue

  WINDOW=$(echo "$agent"   | jq -r '.window')
  WORKTREE=$(echo "$agent" | jq -r '.worktree')
  OBJECTIVE=$(echo "$agent"| jq -r '.objective')
  STATE=$(echo "$agent"    | jq -r '.state // "running"')
  LAST_HASH=$(echo "$agent"| jq -r '.last_output_hash // ""')
  IDLE_SINCE=$(echo "$agent"| jq -r '.idle_since // 0')

  # Pass-through done/escalated agents
  if [[ "$STATE" == "done" || "$STATE" == "escalated" ]]; then
    UPDATED_AGENTS=$(echo "$UPDATED_AGENTS" | jq --argjson a "$agent" '. + [$a]')
    continue
  fi

  # Classify pane
  CLASSIFICATION=$("$CLASSIFY" "$WINDOW" "$IDLE_THRESHOLD" 2>/dev/null || \
    echo '{"state":"error","reason":"classify failed","pane_cmd":"unknown"}')

  PANE_STATE=$(echo "$CLASSIFICATION" | jq -r '.state')
  PANE_REASON=$(echo "$CLASSIFICATION" | jq -r '.reason')

  # Compute content hash for stuck-detection (only for running agents)
  CURRENT_HASH=""
  if [[ "$PANE_STATE" == "running" ]]; then
    RAW=$(tmux capture-pane -t "$WINDOW" -p 2>/dev/null || echo "")
    CURRENT_HASH=$(echo "$RAW" | tail -20 | md5_hash)
  fi

  NEW_STATE="$STATE"
  NEW_IDLE_SINCE="$IDLE_SINCE"
  ACTION="none"
  REASON="$PANE_REASON"

  case "$PANE_STATE" in
    complete)
      NEW_STATE="complete"
      ACTION="complete"
      ;;
    waiting_approval)
      NEW_STATE="waiting_approval"
      ACTION="approve"
      ;;
    idle)
      # Agent process has exited — needs restart
      NEW_STATE="idle"
      ACTION="kick"
      REASON="agent exited (shell is foreground)"
      ;;
    running)
      # Check if hash has been stable (agent may be stuck mid-task)
      if [ -n "$CURRENT_HASH" ] && [ "$CURRENT_HASH" = "$LAST_HASH" ] && [ "$LAST_HASH" != "" ]; then
        if [ "$IDLE_SINCE" = "0" ] || [ "$IDLE_SINCE" = "null" ]; then
          NEW_IDLE_SINCE=$NOW
        else
          STUCK_DURATION=$(( NOW - IDLE_SINCE ))
          if [ "$STUCK_DURATION" -gt "$IDLE_THRESHOLD" ]; then
            NEW_STATE="stuck"
            ACTION="kick"
            REASON="output unchanged for ${STUCK_DURATION}s (threshold: ${IDLE_THRESHOLD}s)"
          fi
        fi
      else
        # Output changed — reset idle timer
        NEW_STATE="running"
        NEW_IDLE_SINCE=0
      fi
      ;;
    error)
      REASON="classify error: $PANE_REASON"
      ;;
  esac

  # Build updated agent record
  UPDATED_AGENT=$(echo "$agent" | jq \
    --arg state "$NEW_STATE" \
    --arg hash "$CURRENT_HASH" \
    --argjson now "$NOW" \
    --argjson idle_since "$NEW_IDLE_SINCE" \
    '.state = $state
     | .last_output_hash = (if $hash == "" then .last_output_hash else $hash end)
     | .last_seen_at = $now
     | .idle_since = $idle_since')

  UPDATED_AGENTS=$(echo "$UPDATED_AGENTS" | jq --argjson a "$UPDATED_AGENT" '. + [$a]')

  # Add action if needed
  if [ "$ACTION" != "none" ]; then
    ACTION_OBJ=$(jq -n \
      --arg window "$WINDOW" \
      --arg action "$ACTION" \
      --arg state "$NEW_STATE" \
      --arg worktree "$WORKTREE" \
      --arg objective "$OBJECTIVE" \
      --arg reason "$REASON" \
      '{window:$window, action:$action, state:$state, worktree:$worktree, objective:$objective, reason:$reason}')
    ACTIONS=$(echo "$ACTIONS" | jq --argjson a "$ACTION_OBJ" '. + [$a]')
  fi

done <<< "$AGENTS_JSON"

# Atomic state file update
jq --argjson agents "$UPDATED_AGENTS" \
   --argjson now "$NOW" \
   '.agents = $agents | .last_poll_at = $now' \
   "$STATE_FILE" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "$STATE_FILE"

echo "$ACTIONS"
