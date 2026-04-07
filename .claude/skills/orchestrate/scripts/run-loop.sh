#!/usr/bin/env bash
# run-loop.sh — Mechanical babysitter for the agent fleet (runs in its own tmux window)
#
# Handles ONLY things that need no intelligence:
#   idle    → restart claude (agent process crashed/exited)
#   approve → auto-approve safe dialogs, press Enter on numbered-option dialogs
#   complete → recycle worktree via recycle-agent.sh (after verify-complete.sh passes)
#
# Also monitors:
#   supervisor → restart supervisor Claude window if it exits
#   all-done   → call notify.sh when every agent reaches done/escalated
#
# Stuck agents and task deviation are handled by the supervisor (dedicated Claude
# window) which has real context and can give targeted guidance.
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
POLL_INTERVAL="${POLL_INTERVAL:-30}"
REBRIEFED_COOLDOWN=300  # 5 minutes between re-briefs for the same agent

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
# handle_kick WINDOW STATE — only for idle (crashed) agents, not stuck
# ---------------------------------------------------------------------------
handle_kick() {
  local window="$1" state="$2"
  [[ "$state" != "idle" ]] && return  # stuck agents handled by supervisor

  local worktree_path objective
  worktree_path=$(agent_field "$window" "worktree_path")
  objective=$(agent_field "$window" "objective")

  echo "[$(date +%H:%M:%S)] KICK restart  $window — agent exited, restarting"
  tmux send-keys -t "$window" "cd '${worktree_path}' && claude --permission-mode bypassPermissions" Enter

  # Wait up to 60s for claude to be fully interactive: node + ❯ prompt visible
  local kick_prompt_found=false
  for i in $(seq 1 60); do
    local kick_cmd kick_pane
    kick_cmd=$(tmux display-message -t "$window" -p '#{pane_current_command}' 2>/dev/null || echo "")
    kick_pane=$(tmux capture-pane -t "$window" -p 2>/dev/null || echo "")
    if echo "$kick_pane" | grep -q "Enter to confirm"; then
      tmux send-keys -t "$window" Down Enter
      sleep 2
      continue
    fi
    if [[ "$kick_cmd" == "node" ]] && echo "$kick_pane" | grep -q "❯"; then
      kick_prompt_found=true
      break
    fi
    sleep 1
  done

  if ! $kick_prompt_found; then
    echo "[$(date +%H:%M:%S)] KICK WARNING  $window — timed out waiting for ❯, sending objective anyway"
  fi

  tmux send-keys -t "$window" "${objective}. When all work is done, output the exact string: ORCHESTRATOR:DONE"
  sleep 0.3
  tmux send-keys -t "$window" Enter
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
# handle_complete WINDOW — verify before recycling; re-brief with cooldown if not done
# ---------------------------------------------------------------------------
handle_complete() {
  local window="$1"
  local worktree_path spare_branch objective
  worktree_path=$(agent_field "$window" "worktree_path")
  spare_branch=$(agent_field "$window" "spare_branch")
  objective=$(agent_field "$window" "objective")

  # Verify completion: check checkpoints, CI, unresolved threads
  local verify_msg
  if ! verify_msg=$(bash "$SCRIPTS_DIR/verify-complete.sh" "$window" 2>&1); then
    # Enforce re-brief cooldown — don't spam the agent every 30s
    local now last_rebriefed elapsed
    now=$(date +%s)
    last_rebriefed=$(agent_field "$window" "last_rebriefed_at")
    last_rebriefed=${last_rebriefed:-0}
    [[ "$last_rebriefed" == "null" || -z "$last_rebriefed" ]] && last_rebriefed=0
    elapsed=$(( now - last_rebriefed ))

    if [ "$elapsed" -lt "$REBRIEFED_COOLDOWN" ]; then
      local wait_secs=$(( REBRIEFED_COOLDOWN - elapsed ))
      echo "[$(date +%H:%M:%S)] RE-BRIEF COOLDOWN $window — next in ${wait_secs}s ($verify_msg)"
      return
    fi

    echo "[$(date +%H:%M:%S)] NOT DONE       $window — $verify_msg"
    echo "[$(date +%H:%M:%S)] RE-BRIEFING    $window — sending task context from state file"
    local pr_number
    pr_number=$(agent_field "$window" "pr_number")
    tmux send-keys -t "$window" "You output ORCHESTRATOR:DONE but work is not complete: $verify_msg. Re-read your task: cat ~/.claude/orchestrator-state.json | jq '.agents[] | select(.window==\"$window\")' and gh pr view $pr_number --json title,body to reorient. Fix what is missing, then output ORCHESTRATOR:DONE again."
    sleep 0.3
    tmux send-keys -t "$window" Enter
    update_state "$window" "state" "running"
    update_state_int "$window" "last_rebriefed_at" "$(date +%s)"
    return
  fi

  echo "[$(date +%H:%M:%S)] COMPLETE ✓     $window — $verify_msg"
  echo "[$(date +%H:%M:%S)] RECYCLING      $window → $spare_branch"
  bash "$SCRIPTS_DIR/recycle-agent.sh" "$window" "$worktree_path" "$spare_branch"
  update_state "$window" "state" "done"

  # Notify on every completion; all-done check runs in the main loop
  bash "$SCRIPTS_DIR/notify.sh" "✓ Agent $window complete — recycled to $spare_branch" || true
}

# ---------------------------------------------------------------------------
# check_all_done — notify and optionally stop when every agent is terminal
# ---------------------------------------------------------------------------
NOTIFIED_DONE=false

check_all_done() {
  $NOTIFIED_DONE && return  # already notified this run

  local total running
  total=$(jq '.agents | length' "$STATE_FILE" 2>/dev/null || echo 0)
  [ "$total" -eq 0 ] && return

  running=$(jq '[.agents[] | select(.state | test("running|stuck|waiting_approval|idle"))] | length' \
    "$STATE_FILE" 2>/dev/null || echo 1)

  if [ "$running" -eq 0 ]; then
    local done_count escalated_count
    done_count=$(jq '[.agents[] | select(.state == "done")] | length' "$STATE_FILE" 2>/dev/null || echo 0)
    escalated_count=$(jq '[.agents[] | select(.state == "escalated")] | length' "$STATE_FILE" 2>/dev/null || echo 0)
    echo "[$(date +%H:%M:%S)] ALL DONE ✓     ${done_count} done, ${escalated_count} escalated"
    bash "$SCRIPTS_DIR/notify.sh" "🏁 Fleet complete — ${done_count}/${total} done, ${escalated_count} escalated. Run capacity.sh to see free worktrees." || true
    NOTIFIED_DONE=true
  fi
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M:%S)] run-loop started (mechanical only, poll every ${POLL_INTERVAL}s)"
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
      complete) handle_complete "$WINDOW" || true; DONE=$(( DONE + 1 )) ;;
    esac
  done < <(echo "$ACTIONS" | jq -c '.[]' 2>/dev/null || true)

  # Check if fleet is fully complete
  check_all_done || true

  RUNNING=$(jq '[.agents[] | select(.state | test("running|stuck|waiting_approval|idle"))] | length' \
    "$STATE_FILE" 2>/dev/null || echo 0)

  echo "[$(date +%H:%M:%S)] Poll — ${RUNNING} running  ${KICKED} kicked  ${DONE} recycled"
  sleep "$POLL_INTERVAL"
done
