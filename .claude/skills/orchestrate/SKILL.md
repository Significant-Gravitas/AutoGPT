---
name: orchestrate
description: "Meta-agent supervisor that manages a fleet of Claude Code agents running in tmux windows. Auto-discovers spare worktrees (spare/N branches), spawns agents into new windows, monitors state, kicks idle agents, auto-approves safe confirmations, and recycles worktrees when done. TRIGGER when user asks to supervise agents, manage a fleet, monitor tmux agents, or orchestrate parallel worktrees."
user-invocable: true
argument-hint: "[start|stop|status|add|poll|capacity] — start spawns agents into spare worktrees, add assigns one more task, capacity shows available worktrees, poll runs one check cycle"
metadata:
  author: autogpt-team
  version: "2.0.0"
---

# Orchestrate — Agent Fleet Supervisor

One tmux session, N windows — each window is one agent in one worktree. The orchestrator auto-discovers spare worktrees, spawns agents, monitors them, and recycles worktrees when work is done. No manual window setup required.

## Worktree lifecycle

```
spare/N branch   →   orchestrate add   →   new window + feat/branch + claude running
                                                        ↓
                                               ORCHESTRATOR:DONE
                                                        ↓
                                        kill window + git checkout spare/N
                                                        ↓
                                               spare/N (free again)
```

Windows are always capped by worktree count — no creep. Auto-close on completion is how the orchestrator signals a worktree is free for the next task.

## State file

Lives at `~/.claude/orchestrator-state.json` (outside repo, never committed):

```json
{
  "active": true,
  "tmux_session": "autogpt1",
  "idle_threshold_seconds": 300,
  "cron_job_id": "...",
  "last_poll_at": 1712345678,
  "agents": [
    {
      "window": "autogpt1:3",
      "worktree": "AutoGPT6",
      "worktree_path": "/Users/majdyz/Code/AutoGPT6",
      "spare_branch": "spare/6",
      "branch": "feat/my-feature",
      "objective": "Implement X and open a PR",
      "state": "running",
      "last_output_hash": "",
      "last_seen_at": 0,
      "idle_since": 0,
      "revision_count": 0
    }
  ]
}
```

Agent states: `running` | `idle` | `stuck` | `waiting_approval` | `complete` | `done` | `escalated`

## Scripts

```bash
SKILLS_DIR=$(git rev-parse --show-toplevel)/.claude/skills/orchestrate
POLL_SCRIPT=$SKILLS_DIR/scripts/poll-cycle.sh
CLASSIFY_SCRIPT=$SKILLS_DIR/scripts/classify-pane.sh
STATE_FILE=~/.claude/orchestrator-state.json
```

## find_spare_worktrees — discover available capacity

```bash
# List all worktrees on spare/N branches (these are free to use)
# Output: one line per available worktree: "PATH SPARE_BRANCH"
git worktree list --porcelain | awk '
  /^worktree / { path = substr($0, 10) }
  /^branch /   { branch = substr($0, 8); print path " " branch }
' | grep -E " refs/heads/spare/[0-9]+$" | sed 's|refs/heads/||'
```

Example output:
```
/Users/majdyz/Code/AutoGPT3 spare/3
/Users/majdyz/Code/AutoGPT7 spare/7
```

## spawn_agent — create window, launch agent, send task

```bash
# Usage: spawn_agent SESSION WORKTREE_PATH SPARE_BRANCH NEW_BRANCH OBJECTIVE
# Returns: "SESSION:WINDOW_IDX" on stdout

SESSION="$1"
WORKTREE_PATH="$2"
SPARE_BRANCH="$3"       # e.g. spare/6  — to restore on completion
NEW_BRANCH="$4"         # e.g. feat/my-feature  — branch to create for this task
OBJECTIVE="$5"
WORKTREE_NAME=$(basename "$WORKTREE_PATH")

# Create the task branch
git -C "$WORKTREE_PATH" checkout -b "$NEW_BRANCH" 2>/dev/null \
  || git -C "$WORKTREE_PATH" checkout "$NEW_BRANCH"

# Create a new named window, capture its numeric index
WIN_IDX=$(tmux new-window -t "$SESSION" -n "$WORKTREE_NAME" -P -F '#{window_index}')
WINDOW="${SESSION}:${WIN_IDX}"

# Launch claude with bypass permissions
tmux send-keys -t "$WINDOW" "cd $WORKTREE_PATH && claude --permission-mode bypassPermissions" Enter

# Wait up to 30s for claude to start (foreground becomes 'node')
for i in $(seq 1 30); do
  CMD=$(tmux display-message -t "$WINDOW" -p '#{pane_current_command}' 2>/dev/null || echo "")
  [[ "$CMD" == "node" ]] && break
  sleep 1
done

# Auto-dismiss Claude settings error dialog if present
CMD=$(tmux display-message -t "$WINDOW" -p '#{pane_current_command}' 2>/dev/null || echo "")
if [[ "$CMD" != "node" ]]; then
  PANE=$(tmux capture-pane -t "$WINDOW" -p 2>/dev/null | tail -5)
  if echo "$PANE" | grep -q "Enter to confirm"; then
    tmux send-keys -t "$WINDOW" Down Enter
    sleep 2
  fi
fi

# Send the task
tmux send-keys -t "$WINDOW" "${OBJECTIVE}. When all work is done, output the exact string: ORCHESTRATOR:DONE" Enter

echo "$WINDOW"
```

## recycle_worktree — restore worktree to spare after completion

```bash
# Usage: recycle_worktree WINDOW WORKTREE_PATH SPARE_BRANCH
WINDOW="$1"
WORKTREE_PATH="$2"
SPARE_BRANCH="$3"

# Kill the tmux window
tmux kill-window -t "$WINDOW" 2>/dev/null

# Restore to spare branch (clean working tree first)
git -C "$WORKTREE_PATH" reset --hard HEAD 2>/dev/null
git -C "$WORKTREE_PATH" checkout "$SPARE_BRANCH"

echo "Recycled: $WORKTREE_PATH → $SPARE_BRANCH (window $WINDOW closed)"
```

## Subcommand: capacity

Show available worktrees before starting. Run this to understand the fleet size.

```bash
echo "=== Available (spare) worktrees ==="
git worktree list --porcelain | awk '
  /^worktree / { path = substr($0, 10) }
  /^branch /   { branch = substr($0, 8); print path " " branch }
' | grep -E " refs/heads/spare/[0-9]+$" | sed 's|refs/heads/||' \
  | while read path branch; do
      echo "  ✓ $path ($branch)"
    done

echo ""
echo "=== In-use worktrees ==="
jq -r '.agents[] | select(.state != "done") | "  [\(.state)] \(.worktree_path) → \(.branch)"' \
  ~/.claude/orchestrator-state.json 2>/dev/null || echo "  (no active state file)"
```

## Subcommand: start

Gather task list, auto-assign spare worktrees, spawn agents, start polling.

### Step 1 — Resolve tmux session

If `$ARGUMENTS` provides a session name, use it. Otherwise:

```bash
tmux list-sessions -F "#{session_name}: #{session_windows} windows" 2>/dev/null
```

If no sessions exist, create one:
```bash
tmux new-session -d -s autogpt1
```

### Step 2 — Show available capacity

```bash
git worktree list --porcelain | awk '
  /^worktree / { path = substr($0, 10) }
  /^branch /   { branch = substr($0, 8); print path " " branch }
' | grep -E " refs/heads/spare/[0-9]+$" | sed 's|refs/heads/||'
```

Show the user how many spare worktrees are available.

### Step 3 — Gather tasks

For each task the user wants to run, collect:
- **objective**: what the agent should do
- **branch name**: e.g. `feat/my-feature` (derived from objective if not given)

The worktree is auto-assigned from the spare list — the user does not need to specify it.

Also ask: **idle_threshold_seconds** (default: 300).

### Step 4 — Spawn agents

For each task, pick the next available spare worktree and spawn:

```bash
SPARE_LIST=$(git worktree list --porcelain | awk '
  /^worktree / { path = substr($0, 10) }
  /^branch /   { branch = substr($0, 8); print path " " branch }
' | grep -E " refs/heads/spare/[0-9]+$" | sed 's|refs/heads/||')

AGENTS_JSON="[]"
while IFS= read -r spare_line && [[ -n "$TASK" ]]; do
  WORKTREE_PATH=$(echo "$spare_line" | awk '{print $1}')
  SPARE_BRANCH=$(echo "$spare_line" | awk '{print $2}')
  WORKTREE_NAME=$(basename "$WORKTREE_PATH")

  WINDOW=$(spawn_agent "$SESSION" "$WORKTREE_PATH" "$SPARE_BRANCH" "$NEW_BRANCH" "$OBJECTIVE")

  AGENT=$(jq -n \
    --arg window "$WINDOW" \
    --arg worktree "$WORKTREE_NAME" \
    --arg path "$WORKTREE_PATH" \
    --arg spare "$SPARE_BRANCH" \
    --arg branch "$NEW_BRANCH" \
    --arg obj "$OBJECTIVE" \
    '{window:$window, worktree:$worktree, worktree_path:$path, spare_branch:$spare,
      branch:$branch, objective:$obj, state:"running", last_output_hash:"",
      last_seen_at:0, idle_since:0, revision_count:0}')

  AGENTS_JSON=$(echo "$AGENTS_JSON" | jq --argjson a "$AGENT" '. + [$a]')
  echo "Spawned: $WINDOW ($WORKTREE_NAME on $NEW_BRANCH)"
done <<< "$SPARE_LIST"
```

### Step 5 — Write state file

```bash
jq -n \
  --arg session "$SESSION" \
  --argjson threshold "$THRESHOLD" \
  --argjson agents "$AGENTS_JSON" \
  '{active:true, tmux_session:$session, idle_threshold_seconds:$threshold,
    cron_job_id:null, last_poll_at:0, agents:$agents}' \
  > ~/.claude/orchestrator-state.json
```

### Step 6 — Run first poll, then start CronCreate

```bash
SKILLS_DIR=$(git rev-parse --show-toplevel)/.claude/skills/orchestrate
bash $SKILLS_DIR/scripts/poll-cycle.sh | jq .
```

CronCreate prompt (fill in SKILLS_DIR with absolute path):
```
Orchestrator poll cycle.

Run: bash SKILLS_DIR/scripts/poll-cycle.sh

Read the JSON output. For each action:

- "kick" (idle — shell foreground): get worktree_path + spare_branch from state file.
  Run: tmux send-keys -t <window> "cd <worktree_path> && claude --permission-mode bypassPermissions" Enter
  Wait 3s. If pane shows "Enter to confirm", send Down+Enter first.
  Then send: "<objective>. When done output ORCHESTRATOR:DONE" Enter

- "kick" (stuck — claude still running): send nudge only, do NOT restart:
  tmux send-keys -t <window> "Continue with your task. Review errors. When done output ORCHESTRATOR:DONE" Enter

- "approve": inspect pane tail. Send "y" Enter if safe (git/install/test/docker/localhost curl).
  Escalate (mark state=escalated) for: rm -rf outside worktree, force push, sudo, secrets.
  Settings dialog "Enter to confirm": send Down+Enter

- "complete": mark done, then recycle — kill the window and restore spare branch:
  tmux kill-window -t <window>
  git -C <worktree_path> reset --hard HEAD
  git -C <worktree_path> checkout <spare_branch>
  Update state: .state = "done"
  Output: "AGENT DONE + RECYCLED: <window> (<worktree>) → <spare_branch>"

After all actions: "Poll [HH:MM] — N agents: X running, Y kicked, Z done+recycled, V escalated"
```

Store cron job ID in state file after CronCreate returns.

## Subcommand: add

Assign one new task to the next available spare worktree.

```bash
SESSION=$(jq -r '.tmux_session' ~/.claude/orchestrator-state.json)

# Find first spare worktree
SPARE_LINE=$(git worktree list --porcelain | awk '
  /^worktree / { path = substr($0, 10) }
  /^branch /   { branch = substr($0, 8); print path " " branch }
' | grep -E " refs/heads/spare/[0-9]+$" | sed 's|refs/heads/||' | head -1)

if [ -z "$SPARE_LINE" ]; then
  echo "No spare worktrees available. All worktrees are in use."
  echo "Wait for a task to complete, or check /orchestrate capacity."
  exit 1
fi

WORKTREE_PATH=$(echo "$SPARE_LINE" | awk '{print $1}')
SPARE_BRANCH=$(echo "$SPARE_LINE" | awk '{print $2}')
WORKTREE_NAME=$(basename "$WORKTREE_PATH")

WINDOW=$(spawn_agent "$SESSION" "$WORKTREE_PATH" "$SPARE_BRANCH" "$NEW_BRANCH" "$OBJECTIVE")

NEW_AGENT=$(jq -n \
  --arg window "$WINDOW" \
  --arg worktree "$WORKTREE_NAME" \
  --arg path "$WORKTREE_PATH" \
  --arg spare "$SPARE_BRANCH" \
  --arg branch "$NEW_BRANCH" \
  --arg obj "$OBJECTIVE" \
  '{window:$window, worktree:$worktree, worktree_path:$path, spare_branch:$spare,
    branch:$branch, objective:$obj, state:"running", last_output_hash:"",
    last_seen_at:0, idle_since:0, revision_count:0}')

jq --argjson a "$NEW_AGENT" '.agents += [$a]' ~/.claude/orchestrator-state.json > /tmp/orch.tmp \
  && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json

echo "Added: $WINDOW ($WORKTREE_NAME on $NEW_BRANCH)"
```

## Subcommand: stop

```bash
# Show state
jq '.agents[] | {window, state, worktree, branch}' ~/.claude/orchestrator-state.json

# Mark inactive
jq '.active = false' ~/.claude/orchestrator-state.json > /tmp/orch.tmp \
  && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json

# Cancel cron
CRON_ID=$(jq -r '.cron_job_id // ""' ~/.claude/orchestrator-state.json)
# Use CronDelete with CRON_ID (skip if empty)
```

Does NOT recycle worktrees — agents may still be mid-task. Use `/orchestrate capacity` to see what's still running.

## Subcommand: status

```bash
jq -r '
  "=== Orchestrator [\(if .active then "RUNNING" else "STOPPED" end)] ===",
  "Session: \(.tmux_session)  |  Idle threshold: \(.idle_threshold_seconds)s",
  "Last poll: \(if .last_poll_at == 0 then "never" else (.last_poll_at | strftime("%H:%M:%S")) end)",
  "",
  (.agents[] | "  [\(.state | ascii_upcase)] \(.window)  \(.worktree)/\(.branch)\n    \(.objective | .[0:70])")
' ~/.claude/orchestrator-state.json

for WINDOW in $(jq -r '.agents[] | select(.state != "done") | .window' ~/.claude/orchestrator-state.json); do
  CMD=$(tmux display-message -t "$WINDOW" -p '#{pane_current_command}' 2>/dev/null || echo "unreachable")
  echo "  $WINDOW live: $CMD"
done
```

## Subcommand: poll

Manual poll cycle (same logic as the CronCreate loop).

```bash
SKILLS_DIR=$(git rev-parse --show-toplevel)/.claude/skills/orchestrate
ACTIONS=$(bash $SKILLS_DIR/scripts/poll-cycle.sh)
echo "$ACTIONS" | jq .
```

Process each action per the rules in the CronCreate prompt above.

## Key rules

1. **Auto-assign spare worktrees** — never ask the user to pick a worktree path manually
2. **Auto-close + recycle on done** — kill window, restore `spare/N` branch; this is how capacity frees up
3. **Never restart a running agent** — only restart when foreground process is a shell
4. **Handle settings errors automatically** — if "Enter to confirm" appears, send Down+Enter
5. **Always use `--permission-mode bypassPermissions`** on every spawn
6. **Escalate after 3 kicks** — mark as `escalated`, alert user
7. **Atomic state writes** — `.tmp` + `mv` always
8. **Never approve destructive commands** — when in doubt, escalate
9. **Loop is session-scoped** — stops when this Claude session closes
