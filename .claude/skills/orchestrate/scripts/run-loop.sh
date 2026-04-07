#!/usr/bin/env bash
# run-loop.sh — Mechanical babysitter for the agent fleet (runs in its own tmux window)
#
# Handles ONLY things that need no intelligence:
#   idle    → restart claude (agent process crashed/exited)
#   approve → auto-approve safe dialogs, press Enter on numbered-option dialogs
#   complete → verify + mark done + notify (NO auto-recycle — user must explicitly recycle)
#
# Worktrees are NEVER recycled automatically. Recycling requires explicit user consent.
# To recycle a completed window: bash recycle-agent.sh WINDOW WORKTREE_PATH SPARE_BRANCH
# Sessions can be resumed at any time: claude --resume SESSION_ID --permission-mode bypassPermissions
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
# handle_kick WINDOW STATE — only for idle (crashed) agents, not stuck
# ---------------------------------------------------------------------------
handle_kick() {
  local window="$1" state="$2"
  [[ "$state" != "idle" ]] && return  # stuck agents handled by supervisor

  local worktree_path session_id
  worktree_path=$(agent_field "$window" "worktree_path")
  session_id=$(agent_field "$window" "session_id")

  echo "[$(date +%H:%M:%S)] KICK restart  $window — agent exited, resuming session"

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
# handle_complete WINDOW — verify; re-brief if not done; mark done+notify (NO recycle)
# ---------------------------------------------------------------------------
handle_complete() {
  local window="$1"
  local objective
  objective=$(agent_field "$window" "objective")

  # Verify completion: checkpoints, CI, unresolved threads, CHANGES_REQUESTED
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

  # Verified complete — mark done and notify. DO NOT recycle the window.
  # The worktree stays on its branch, the session stays alive, the window stays open.
  # Recycling requires explicit user/orchestrator consent.
  echo "[$(date +%H:%M:%S)] COMPLETE ✓     $window — $verify_msg"
  echo "[$(date +%H:%M:%S)] MARKED DONE    $window — window kept open; recycle when ready"
  update_state "$window" "state" "done"

  local worktree spare_branch session_id
  worktree=$(agent_field "$window" "worktree")
  spare_branch=$(agent_field "$window" "spare_branch")
  session_id=$(agent_field "$window" "session_id")

  local resume_hint=""
  [ -n "$session_id" ] && resume_hint=" Resume: claude --resume $session_id --permission-mode bypassPermissions"

  bash "$SCRIPTS_DIR/notify.sh" "✓ $window ($worktree) done — window kept open.${resume_hint} To recycle: recycle-agent.sh $window WORKTREE_PATH $spare_branch" || true
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
