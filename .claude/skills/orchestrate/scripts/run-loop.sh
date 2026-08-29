#!/usr/bin/env bash
# run-loop.sh — Mechanical babysitter for the agent fleet (runs in its own tmux window)
#
# Handles ONLY two things that need no intelligence:
#   idle    → restart claude using --resume SESSION_ID (or --continue) to restore context
#   approve → auto-approve safe dialogs, press Enter on numbered-option dialogs
#
# Everything else — ORCHESTRATOR:DONE, verification, /pr-test, final evaluation,
# marking done, deciding to close windows — is the orchestrating Claude's job.
# poll-cycle.sh sets state to pending_evaluation when ORCHESTRATOR:DONE is detected;
# the orchestrator's background poll loop handles it from there.
#
# Usage: run-loop.sh
# Env:   POLL_INTERVAL (default: 30), ORCHESTRATOR_STATE_FILE

set -euo pipefail

# Copy scripts to a stable location outside the repo so they survive branch
# checkouts (e.g. recycle-agent.sh switching spare/N back into this worktree
# would wipe .claude/skills/orchestrate/scripts if the skill only exists on the
# current branch).
_ORIGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STABLE_SCRIPTS_DIR="$HOME/.claude/orchestrator/scripts"
mkdir -p "$STABLE_SCRIPTS_DIR"
cp "$_ORIGIN_DIR"/*.sh "$STABLE_SCRIPTS_DIR/"
chmod +x "$STABLE_SCRIPTS_DIR"/*.sh
SCRIPTS_DIR="$STABLE_SCRIPTS_DIR"

STATE_FILE="${ORCHESTRATOR_STATE_FILE:-$HOME/.claude/orchestrator-state.json}"
# Adaptive polling: starts at base interval, backs off up to POLL_IDLE_MAX when
# no agents need attention, resets on any activity or waiting_approval state.
POLL_INTERVAL="${POLL_INTERVAL:-30}"
POLL_IDLE_MAX=${POLL_IDLE_MAX:-300}
POLL_CURRENT=$POLL_INTERVAL

# ---------------------------------------------------------------------------
# update_state WINDOW FIELD VALUE
# ---------------------------------------------------------------------------
update_state() {
  local window="$1" field="$2" value="$3"
  jq --arg w "$window" --arg f "$field" --arg v "$value" \
    '.agents |= map(if .window == $w then .[$f] = $v else . end)' \
    "$STATE_FILE" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

update_state_int() {
  local window="$1" field="$2" value="$3"
  jq --arg w "$window" --arg f "$field" --argjson v "$value" \
    '.agents |= map(if .window == $w then .[$f] = $v else . end)' \
    "$STATE_FILE" > "${STATE_FILE}.tmp" && mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

agent_field() {
  jq -r --arg w "$1" --arg f "$2" \
    '.agents[] | select(.window == $w) | .[$f] // ""' \
    "$STATE_FILE" 2>/dev/null
}

# ---------------------------------------------------------------------------
# wait_for_prompt WINDOW — wait up to 60s for Claude's ❯ prompt
# ---------------------------------------------------------------------------
wait_for_prompt() {
  local window="$1"
  for i in $(seq 1 60); do
    local cmd pane
    cmd=$(tmux display-message -t "$window" -p '#{pane_current_command}' 2>/dev/null || echo "")
    pane=$(tmux capture-pane -t "$window" -p 2>/dev/null || echo "")
    if echo "$pane" | grep -q "Enter to confirm"; then
      tmux send-keys -t "$window" Down Enter; sleep 2; continue
    fi
    [[ "$cmd" == "node" ]] && echo "$pane" | grep -q "❯" && return 0
    sleep 1
  done
  return 1  # timed out
}

# ---------------------------------------------------------------------------
# wait_for_claude_idle WINDOW — wait up to 30s for Claude to reach idle ❯ prompt
# (no spinner or busy indicator visible in the last 3 lines of pane output)
# Returns 0 when idle, 1 on timeout.
# ---------------------------------------------------------------------------
wait_for_claude_idle() {
  local window="$1"
  local timeout="${2:-30}"
  local elapsed=0
  while (( elapsed < timeout )); do
    local cmd pane pane_tail
    cmd=$(tmux display-message -t "$window" -p '#{pane_current_command}' 2>/dev/null || echo "")
    pane=$(tmux capture-pane -t "$window" -p 2>/dev/null || echo "")
    pane_tail=$(echo "$pane" | tail -3)
    # Check full pane (not just tail) — 'Enter to confirm' dialog can scroll above last 3 lines.
    # Do NOT reset elapsed — resetting allows an infinite loop if the dialog never clears.
    if echo "$pane" | grep -q "Enter to confirm"; then
      tmux send-keys -t "$window" Down Enter
      sleep 2; (( elapsed += 2 )); continue
    fi
    # Must be running under node (Claude is live)
    if [[ "$cmd" == "node" ]]; then
      # Idle: ❯ prompt visible AND no spinner/busy text in last 3 lines
      if echo "$pane_tail" | grep -q "❯" && \
         ! echo "$pane_tail" | grep -qE '[✳✽✢✶·✻✼✿❋✤]|Running…|Compacting'; then
        return 0
      fi
    fi
    sleep 2
    (( elapsed += 2 ))
  done
  return 1  # timed out
}

# ---------------------------------------------------------------------------
# handle_kick WINDOW STATE — only for idle (crashed) agents, not stuck
# ---------------------------------------------------------------------------
handle_kick() {
  local window="$1" state="$2"
  [[ "$state" != "idle" ]] && return  # stuck agents handled by supervisor

  local worktree_path session_id
  worktree_path=$(agent_field "$window" "worktree_path")
  session_id=$(agent_field "$window" "session_id")

  echo "[$(date +%H:%M:%S)] KICK restart  $window — agent exited, resuming session"

  # Wait for the shell prompt before typing — avoids sending into a still-draining pane
  wait_for_claude_idle "$window" 30 \
    || echo "[$(date +%H:%M:%S)] KICK WARNING  $window — pane still busy before resume, sending anyway"

  # Resume the exact session so the agent retains full context — no need to re-send objective
  if [ -n "$session_id" ]; then
    tmux send-keys -t "$window" "cd '${worktree_path}' && claude --resume '${session_id}' --permission-mode bypassPermissions" Enter
  else
    tmux send-keys -t "$window" "cd '${worktree_path}' && claude --continue --permission-mode bypassPermissions" Enter
  fi

  wait_for_prompt "$window" || echo "[$(date +%H:%M:%S)] KICK WARNING  $window — timed out waiting for ❯"
}

# ---------------------------------------------------------------------------
# handle_approve WINDOW — auto-approve dialogs that need no judgment
# ---------------------------------------------------------------------------
handle_approve() {
  local window="$1"
  local pane_tail
  pane_tail=$(tmux capture-pane -t "$window" -p 2>/dev/null | tail -3 || echo "")

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
# Main loop
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] run-loop started (mechanical only, poll ${POLL_INTERVAL}s→${POLL_IDLE_MAX}s adaptive)"
echo "[$(date +%H:%M:%S)] Supervisor: orchestrating Claude session (not a separate window)"
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
      complete) DONE=$(( DONE + 1 )) ;;  # poll-cycle already set state=pending_evaluation; orchestrator handles
    esac
  done < <(echo "$ACTIONS" | jq -c '.[]' 2>/dev/null || true)

  RUNNING=$(jq '[.agents[] | select(.state | test("running|stuck|waiting_approval|idle"))] | length' \
    "$STATE_FILE" 2>/dev/null || echo 0)

  # Adaptive backoff: reset to base on activity or waiting_approval agents; back off when truly idle
  WAITING=$(jq '[.agents[] | select(.state == "waiting_approval")] | length' "$STATE_FILE" 2>/dev/null || echo 0)
  if (( KICKED > 0 || DONE > 0 || WAITING > 0 )); then
    POLL_CURRENT=$POLL_INTERVAL
  else
    POLL_CURRENT=$(( POLL_CURRENT + POLL_CURRENT / 2 + 1 ))
    (( POLL_CURRENT > POLL_IDLE_MAX )) && POLL_CURRENT=$POLL_IDLE_MAX
  fi

  echo "[$(date +%H:%M:%S)] Poll — ${RUNNING} running  ${KICKED} kicked  ${DONE} recycled  (next in ${POLL_CURRENT}s)"
  sleep "$POLL_CURRENT"
done
