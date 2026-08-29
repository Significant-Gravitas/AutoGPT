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

set -euo pipefail

STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFY="$SCRIPTS_DIR/classify-pane.sh"

# Cross-platform md5: always outputs just the hex digest
md5_hash() {
  if command -v md5sum &>/dev/null; then
    md5sum | awk '{print $1}'
  else
    md5 | awk '{print $NF}'
  fi
}

# Clean up temp file on any exit (avoids stale .tmp if jq write fails)
trap 'rm -f "${STATE_FILE}.tmp"' EXIT

# Ensure state file exists
if [ ! -f "$STATE_FILE" ]; then
  echo '{"active":false,"agents":[]}' > "$STATE_FILE"
fi

# Validate JSON upfront before any jq reads that run under set -e.
# A truncated/corrupt file (e.g. from a SIGKILL mid-write) would otherwise
# abort the script at the ACTIVE read below without emitting any JSON output.
if ! jq -e '.' "$STATE_FILE" >/dev/null 2>&1; then
  echo "State file parse error — check $STATE_FILE" >&2
  echo "[]"
  exit 0
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

# Read agents as newline-delimited JSON objects.
# jq exits non-zero when .agents[] has no matches on an empty array, which is valid —
# so we suppress that exit code and separately validate the file is well-formed JSON.
if ! AGENTS_JSON=$(jq -e -c '.agents // empty | .[]' "$STATE_FILE" 2>/dev/null); then
  if ! jq -e '.' "$STATE_FILE" > /dev/null 2>&1; then
    echo "State file parse error — check $STATE_FILE" >&2
  fi
  echo "[]"
  exit 0
fi

if [ -z "$AGENTS_JSON" ]; then
  echo "[]"
  exit 0
fi

while IFS= read -r agent; do
  [ -z "$agent" ] && continue

  # Use // "" defaults so a single malformed field doesn't abort the whole cycle
  WINDOW=$(echo "$agent"   | jq -r '.window // ""')
  WORKTREE=$(echo "$agent" | jq -r '.worktree // ""')
  OBJECTIVE=$(echo "$agent"| jq -r '.objective // ""')
  STATE=$(echo "$agent"    | jq -r '.state // "running"')
  LAST_HASH=$(echo "$agent"| jq -r '.last_output_hash // ""')
  IDLE_SINCE=$(echo "$agent"| jq -r '.idle_since // 0')
  REVISION_COUNT=$(echo "$agent"| jq -r '.revision_count // 0')

  # Validate window format to prevent tmux target injection.
  # Allow session:window (numeric or named) and session:window.pane
  if ! [[ "$WINDOW" =~ ^[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+(\.[0-9]+)?$ ]]; then
    echo "Skipping agent with invalid window value: $WINDOW" >&2
    UPDATED_AGENTS=$(echo "$UPDATED_AGENTS" | jq --argjson a "$agent" '. + [$a]')
    continue
  fi

  # Pass-through terminal-state agents
  if [[ "$STATE" == "done" || "$STATE" == "escalated" || "$STATE" == "complete" || "$STATE" == "pending_evaluation" ]]; then
    UPDATED_AGENTS=$(echo "$UPDATED_AGENTS" | jq --argjson a "$agent" '. + [$a]')
    continue
  fi

  # Classify pane.
  # classify-pane.sh always emits JSON before exit (even on error), so using
  # "|| echo '...'" would concatenate two JSON objects when it exits non-zero.
  # Use "|| true" inside the substitution so set -euo pipefail does not abort
  # the poll cycle when classify exits with a non-zero status code.
  CLASSIFICATION=$("$CLASSIFY" "$WINDOW" 2>/dev/null || true)
  [ -z "$CLASSIFICATION" ] && CLASSIFICATION='{"state":"error","reason":"classify failed","pane_cmd":"unknown"}'

  PANE_STATE=$(echo "$CLASSIFICATION" | jq -r '.state')
  PANE_REASON=$(echo "$CLASSIFICATION" | jq -r '.reason')

  # Capture full pane output once — used for hash (stuck detection) and checkpoint parsing.
  # Use -S -500 to get the last ~500 lines of scrollback so checkpoints aren't missed.
  RAW=$(tmux capture-pane -t "$WINDOW" -p -S -500 2>/dev/null || echo "")

  # --- Checkpoint tracking ---
  # Parse any "CHECKPOINT:<step>" lines the agent has output and merge into state file.
  # The agent writes these as it completes each required step so verify-complete.sh can gate recycling.
  EXISTING_CPS=$(echo "$agent" | jq -c '.checkpoints // []')
  NEW_CHECKPOINTS_JSON="$EXISTING_CPS"
  if [ -n "$RAW" ]; then
    FOUND_CPS=$(echo "$RAW" \
      | grep -oE "CHECKPOINT:[a-zA-Z0-9_-]+" \
      | sed 's/CHECKPOINT://' \
      | sort -u \
      | jq -R . | jq -s . 2>/dev/null || echo "[]")
    NEW_CHECKPOINTS_JSON=$(jq -n \
      --argjson existing "$EXISTING_CPS" \
      --argjson found "$FOUND_CPS" \
      '($existing + $found) | unique' 2>/dev/null || echo "$EXISTING_CPS")
  fi

  # Compute content hash for stuck-detection (only for running agents)
  CURRENT_HASH=""
  if [[ "$PANE_STATE" == "running" ]] && [ -n "$RAW" ]; then
    CURRENT_HASH=$(echo "$RAW" | tail -20 | md5_hash)
  fi

  NEW_STATE="$STATE"
  NEW_IDLE_SINCE="$IDLE_SINCE"
  NEW_REVISION_COUNT="$REVISION_COUNT"
  ACTION="none"
  REASON="$PANE_REASON"

  case "$PANE_STATE" in
    complete)
      # Agent output ORCHESTRATOR:DONE — mark pending_evaluation so orchestrator handles it.
      # run-loop does NOT verify or notify; orchestrator's background poll picks this up.
      NEW_STATE="pending_evaluation"
      ACTION="complete"  # run-loop logs it but takes no action
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
      NEW_REVISION_COUNT=$(( REVISION_COUNT + 1 ))
      NEW_IDLE_SINCE=$NOW
      if [ "$NEW_REVISION_COUNT" -ge 3 ]; then
        NEW_STATE="escalated"
        ACTION="none"
        REASON="escalated after ${NEW_REVISION_COUNT} kicks — needs human attention"
      fi
      ;;
    running)
      # Clear idle_since only when transitioning from idle (agent was kicked and
      # restarted). Do NOT reset for stuck — idle_since must persist across polls
      # so STUCK_DURATION can accumulate and trigger escalation.
      # Also update the local IDLE_SINCE so the hash-stability check below uses
      # the reset value on this same poll, not the stale kick timestamp.
      if [[ "$STATE" == "idle" ]]; then
        NEW_IDLE_SINCE=0
        IDLE_SINCE=0
      fi
      # Check if hash has been stable (agent may be stuck mid-task)
      if [ -n "$CURRENT_HASH" ] && [ "$CURRENT_HASH" = "$LAST_HASH" ] && [ "$LAST_HASH" != "" ]; then
        if [ "$IDLE_SINCE" = "0" ] || [ "$IDLE_SINCE" = "null" ]; then
          NEW_IDLE_SINCE=$NOW
        else
          STUCK_DURATION=$(( NOW - IDLE_SINCE ))
          if [ "$STUCK_DURATION" -gt "$IDLE_THRESHOLD" ]; then
            NEW_REVISION_COUNT=$(( REVISION_COUNT + 1 ))
            NEW_IDLE_SINCE=$NOW
            if [ "$NEW_REVISION_COUNT" -ge 3 ]; then
              NEW_STATE="escalated"
              ACTION="none"
              REASON="escalated after ${NEW_REVISION_COUNT} kicks — needs human attention"
            else
              NEW_STATE="stuck"
              ACTION="kick"
              REASON="output unchanged for ${STUCK_DURATION}s (threshold: ${IDLE_THRESHOLD}s)"
            fi
          fi
        fi
      else
        # Only reset the idle timer when we have a valid hash comparison (pane
        # capture succeeded). If CURRENT_HASH is empty (tmux capture-pane failed),
        # preserve existing timers so stuck detection is not inadvertently reset.
        if [ -n "$CURRENT_HASH" ]; then
          NEW_STATE="running"
          NEW_IDLE_SINCE=0
        fi
      fi
      ;;
    error)
      REASON="classify error: $PANE_REASON"
      ;;
  esac

  # Build updated agent record (ensure idle_since and revision_count are numeric)
  # Use || true on each jq call so a malformed field skips this agent rather than
  # aborting the entire poll cycle under set -e.
  UPDATED_AGENT=$(echo "$agent" | jq \
    --arg state "$NEW_STATE" \
    --arg hash "$CURRENT_HASH" \
    --argjson now "$NOW" \
    --arg idle_since "$NEW_IDLE_SINCE" \
    --arg revision_count "$NEW_REVISION_COUNT" \
    --argjson checkpoints "$NEW_CHECKPOINTS_JSON" \
    '.state = $state
     | .last_output_hash = (if $hash == "" then .last_output_hash else $hash end)
     | .last_seen_at = $now
     | .idle_since = ($idle_since | tonumber)
     | .revision_count = ($revision_count | tonumber)
     | .checkpoints = $checkpoints' 2>/dev/null) || {
    echo "Warning: failed to build updated agent for window $WINDOW — keeping original" >&2
    UPDATED_AGENTS=$(echo "$UPDATED_AGENTS" | jq --argjson a "$agent" '. + [$a]')
    continue
  }

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
