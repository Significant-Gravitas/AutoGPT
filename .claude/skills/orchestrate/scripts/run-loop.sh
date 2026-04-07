#!/usr/bin/env bash
# run-loop.sh — Mechanical babysitter for the agent fleet (runs in its own tmux window)
#
# Handles ONLY things that need no intelligence:
#   idle    → restart claude (agent process crashed/exited)
#   approve → auto-approve safe dialogs, press Enter on numbered-option dialogs
#   complete → recycle worktree via recycle-agent.sh
#
# Stuck agents and task deviation are handled by the supervisor (CronCreate in
# the main Claude session) which has real context and can give targeted guidance.
#
# Usage: run-loop.sh
# Env:   POLL_INTERVAL (default: 30), ORCHESTRATOR_STATE_FILE

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

# ---------------------------------------------------------------------------
# update_state WINDOW FIELD VALUE
# ---------------------------------------------------------------------------
update_state() {
  local window="$1" field="$2" value="$3"
  jq --arg w "$window" --arg f "$field" --arg v "$value" \
    '.agents |= map(if .window == $w then .[$f] = $v else . end)' \
    "$STATE_FILE" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

agent_field() {
  jq -r --arg w "$1" --arg f "$2" \
    '.agents[] | select(.window == $w) | .[$f] // ""' \
    "$STATE_FILE" 2>/dev/null
}

# ---------------------------------------------------------------------------
# handle_kick WINDOW — only for idle (crashed) agents, not stuck
# ---------------------------------------------------------------------------
handle_kick() {
  local window="$1" state="$2"
  [[ "$state" != "idle" ]] && return  # stuck agents handled by supervisor

  local worktree_path objective
  worktree_path=$(agent_field "$window" "worktree_path")
  objective=$(agent_field "$window" "objective")

  echo "[$(date +%H:%M:%S)] KICK restart  $window — agent exited, restarting"
  tmux send-keys -t "$window" "cd '${worktree_path}' && claude --permission-mode bypassPermissions" Enter
  sleep 3

  # Auto-dismiss settings error dialog
  local pane
  pane=$(tmux capture-pane -t "$window" -p 2>/dev/null || echo "")
  if echo "$pane" | grep -q "Enter to confirm"; then
    tmux send-keys -t "$window" Down Enter
    sleep 2
  fi

  tmux send-keys -t "$window" "${objective}. When all work is done, output the exact string: ORCHESTRATOR:DONE" Enter
}

# ---------------------------------------------------------------------------
# handle_approve WINDOW — auto-approve dialogs that need no judgment
# ---------------------------------------------------------------------------
handle_approve() {
  local window="$1"
  local pane_tail
  pane_tail=$(tmux capture-pane -t "$window" -p 2>/dev/null | tail -10 || echo "")

  # Settings error dialog at startup
  if echo "$pane_tail" | grep -q "Enter to confirm"; then
    echo "[$(date +%H:%M:%S)] APPROVE dialog $window — settings error"
    tmux send-keys -t "$window" Down Enter
    return
  fi

  # Numbered-option dialog (e.g. "Do you want to make this edit?")
  # ❯ is already on option 1 (Yes) — Enter confirms it
  if echo "$pane_tail" | grep -qE "❯\s*1\." || echo "$pane_tail" | grep -q "Esc to cancel"; then
    echo "[$(date +%H:%M:%S)] APPROVE edit   $window"
    tmux send-keys -t "$window" "" Enter
    return
  fi

  # y/n prompt for safe operations
  if echo "$pane_tail" | grep -qiE "(^git |^npm |^pnpm |^poetry |^pytest|^docker |^make |^cargo |^pip |^yarn |curl .*(localhost|127\.0\.0\.1))"; then
    echo "[$(date +%H:%M:%S)] APPROVE safe   $window"
    tmux send-keys -t "$window" "y" Enter
    return
  fi

  # Anything else — supervisor handles it, just log
  echo "[$(date +%H:%M:%S)] APPROVE skip   $window — unknown dialog, supervisor will handle"
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
echo "[$(date +%H:%M:%S)] run-loop started (mechanical only, poll every ${POLL_INTERVAL}s)"
echo "[$(date +%H:%M:%S)] Supervisor: CronCreate in main Claude session"
echo "---"

while true; do
  if ! jq -e '.active == true' "$STATE_FILE" >/dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] active=false — exiting."
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
      kick)     handle_kick "$WINDOW" "$STATE" || true; KICKED=$(( KICKED + 1 )) ;;
      approve)  handle_approve "$WINDOW" || true ;;
      complete) handle_complete "$WINDOW" || true; DONE=$(( DONE + 1 )) ;;
    esac
  done < <(echo "$ACTIONS" | jq -c '.[]' 2>/dev/null || true)

  RUNNING=$(jq '[.agents[] | select(.state | test("running|stuck|waiting_approval|idle"))] | length' \
    "$STATE_FILE" 2>/dev/null || echo 0)

  echo "[$(date +%H:%M:%S)] Poll — ${RUNNING} running  ${KICKED} kicked  ${DONE} recycled"
  sleep "$POLL_INTERVAL"
done
