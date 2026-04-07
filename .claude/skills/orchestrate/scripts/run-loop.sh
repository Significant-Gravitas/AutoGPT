#!/usr/bin/env bash
# run-loop.sh — Orchestrator polling loop (runs in its own dedicated tmux window)
#
# Calls poll-cycle.sh every POLL_INTERVAL seconds and handles all actions in bash:
#   kick (idle)  → restart agent (spawn claude + send objective)
#   kick (stuck) → nudge running agent
#   approve      → auto-approve safe ops; escalate destructive ones
#   complete     → recycle via recycle-agent.sh + mark done in state file
#
# Usage: run-loop.sh
# Env:   POLL_INTERVAL (default: 30), ORCHESTRATOR_STATE_FILE

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

# Auto-approve: command matches one of these patterns (case-insensitive, extended regex)
SAFE_PATTERNS="(^git |^npm |^pnpm |^poetry |^pytest|^docker |^make |^cargo |^pip |^yarn |^npx |curl .*(localhost|127\.0\.0\.1))"

# Escalate immediately: never auto-approve these
ESCALATE_PATTERNS="(rm -rf [^.~/]|--force.*(main|master)|sudo |GITHUB_TOKEN|API_KEY|SECRET)"

# ---------------------------------------------------------------------------
# update_state WINDOW FIELD VALUE — atomically update one agent field
# ---------------------------------------------------------------------------
update_state() {
  local window="$1" field="$2" value="$3"
  jq --arg w "$window" --arg f "$field" --arg v "$value" \
    '.agents |= map(if .window == $w then .[$f] = $v else . end)' \
    "$STATE_FILE" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

# ---------------------------------------------------------------------------
# agent_field WINDOW FIELD — read a field from state file for one agent
# ---------------------------------------------------------------------------
agent_field() {
  jq -r --arg w "$1" --arg f "$2" \
    '.agents[] | select(.window == $w) | .[$f] // ""' \
    "$STATE_FILE" 2>/dev/null
}

# ---------------------------------------------------------------------------
# handle_kick WINDOW STATE
# ---------------------------------------------------------------------------
handle_kick() {
  local window="$1" state="$2"
  local worktree_path objective

  worktree_path=$(agent_field "$window" "worktree_path")
  objective=$(agent_field "$window" "objective")

  if [[ "$state" == "idle" ]]; then
    echo "[$(date +%H:%M:%S)] KICK restart  $window"
    tmux send-keys -t "$window" "cd '${worktree_path}' && claude --permission-mode bypassPermissions" Enter
    sleep 3
    # Auto-dismiss settings error dialog if present
    local pane
    pane=$(tmux capture-pane -t "$window" -p 2>/dev/null || echo "")
    if echo "$pane" | grep -q "Enter to confirm"; then
      tmux send-keys -t "$window" Down Enter
      sleep 2
    fi
    tmux send-keys -t "$window" "${objective}. When all work is done, output the exact string: ORCHESTRATOR:DONE" Enter
  else
    # stuck — claude is running but output is frozen, nudge it
    echo "[$(date +%H:%M:%S)] KICK nudge    $window (stuck)"
    tmux send-keys -t "$window" "Continue with your task. Review any errors and proceed. When done output ORCHESTRATOR:DONE" Enter
  fi
}

# ---------------------------------------------------------------------------
# handle_approve WINDOW
# ---------------------------------------------------------------------------
handle_approve() {
  local window="$1"
  local pane_tail
  pane_tail=$(tmux capture-pane -t "$window" -p 2>/dev/null | tail -10 || echo "")

  # Settings error dialog (startup)
  if echo "$pane_tail" | grep -q "Enter to confirm"; then
    echo "[$(date +%H:%M:%S)] APPROVE dialog $window — dismissing settings error"
    tmux send-keys -t "$window" Down Enter
    return
  fi

  # Numbered-option dialog (e.g. "Do you want to make this edit?" / "1. Yes / 2. Yes, and... / 3. No")
  # The ❯ cursor is already on option 1 (Yes) — pressing Enter alone confirms it.
  # Sending "1" or "2" types the digit into the input rather than selecting the option.
  if echo "$pane_tail" | grep -qE "❯\s*1\." || echo "$pane_tail" | grep -q "Esc to cancel"; then
    echo "[$(date +%H:%M:%S)] APPROVE numbered $window — pressing Enter (Yes)"
    tmux send-keys -t "$window" "" Enter
    return
  fi

  # Escalation patterns — never auto-approve
  if echo "$pane_tail" | grep -qiE "$ESCALATE_PATTERNS"; then
    echo "[$(date +%H:%M:%S)] ESCALATE       $window — destructive pattern, needs human review:"
    echo "$pane_tail" | tail -5 | sed 's/^/    /'
    update_state "$window" "state" "escalated"
    printf '\a'  # terminal bell
    return
  fi

  # Safe patterns — auto-approve with y
  if echo "$pane_tail" | grep -qiE "$SAFE_PATTERNS"; then
    echo "[$(date +%H:%M:%S)] APPROVE auto   $window"
    tmux send-keys -t "$window" "y" Enter
    return
  fi

  # Unknown — escalate to be safe
  echo "[$(date +%H:%M:%S)] ESCALATE       $window — unknown approval request:"
  echo "$pane_tail" | tail -5 | sed 's/^/    /'
  update_state "$window" "state" "escalated"
  printf '\a'
}

# ---------------------------------------------------------------------------
# handle_complete WINDOW
# ---------------------------------------------------------------------------
handle_complete() {
  local window="$1"
  local worktree_path spare_branch
  worktree_path=$(agent_field "$window" "worktree_path")
  spare_branch=$(agent_field "$window" "spare_branch")

  echo "[$(date +%H:%M:%S)] COMPLETE       $window — recycling to $spare_branch"
  bash "$SCRIPTS_DIR/recycle-agent.sh" "$window" "$worktree_path" "$spare_branch"
  update_state "$window" "state" "done"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] Orchestrator loop started (poll every ${POLL_INTERVAL}s)"
echo "[$(date +%H:%M:%S)] State: $STATE_FILE"
echo "---"

while true; do
  # Stop if orchestrator was deactivated
  if ! jq -e '.active == true' "$STATE_FILE" >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] active=false — loop exiting."
    exit 0
  fi

  ACTIONS=$("$SCRIPTS_DIR/poll-cycle.sh" 2>/dev/null || echo "[]")
  KICKED=0; DONE=0

  while IFS= read -r action; do
    [ -z "$action" ] && continue

    WINDOW=$(echo "$action" | jq -r '.window // ""')
    ACTION=$(echo "$action" | jq -r '.action // ""')
    STATE=$(echo "$action"  | jq -r '.state // ""')

    case "$ACTION" in
      kick)
        handle_kick "$WINDOW" "$STATE" || true
        KICKED=$(( KICKED + 1 ))
        ;;
      approve)
        handle_approve "$WINDOW" || true
        ;;
      complete)
        handle_complete "$WINDOW" || true
        DONE=$(( DONE + 1 ))
        ;;
    esac
  done < <(echo "$ACTIONS" | jq -c '.[]' 2>/dev/null || true)

  RUNNING=$(jq '[.agents[] | select(.state | test("running|stuck|waiting_approval|idle"))] | length' \
    "$STATE_FILE" 2>/dev/null || echo 0)
  ESCALATED=$(jq '[.agents[] | select(.state == "escalated")] | length' \
    "$STATE_FILE" 2>/dev/null || echo 0)

  echo "[$(date +%H:%M:%S)] Poll — ${RUNNING} running  ${KICKED} kicked  ${DONE} recycled  ${ESCALATED} escalated"

  sleep "$POLL_INTERVAL"
done
