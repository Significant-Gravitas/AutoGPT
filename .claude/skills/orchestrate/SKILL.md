---
name: orchestrate
description: "Meta-agent supervisor that manages a fleet of Claude Code agents running in tmux windows. Monitors agent states, auto-approves safe tool calls, kicks idle/exited agents back to work, and reports completions. TRIGGER when user asks to supervise agents, manage a fleet, monitor tmux agents, or orchestrate parallel worktrees."
user-invocable: true
argument-hint: "[start|stop|status|add|poll] — start begins monitoring, stop halts it, status shows fleet, add registers a new agent, poll runs one check cycle manually"
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Orchestrate — Agent Fleet Supervisor

Manages multiple Claude Code agents running in tmux windows. Monitors their state, kicks them when idle, and auto-approves safe confirmations so they never block on human input.

## Setup agents for orchestration

Before registering an agent with the orchestrator, launch it in a tmux window with bypass permissions:

```bash
# In your tmux session (e.g., session named "work"):
# window 0: cd /Users/majdyz/Code/AutoGPT6 && claude --permission-mode bypassPermissions
# window 1: cd /Users/majdyz/Code/AutoGPT9 && claude --permission-mode bypassPermissions
```

Also add this to the end of each agent's initial prompt so the orchestrator knows when it's done:
> "When you have finished all work, output the exact string: ORCHESTRATOR:DONE"

## State file

All state lives at `~/.claude/orchestrator-state.json`. It is NOT in the repo. Schema:

```json
{
  "active": true,
  "tmux_session": "work",
  "idle_threshold_seconds": 300,
  "cron_job_id": "...",
  "last_poll_at": 1712345678,
  "agents": [
    {
      "window": "work:0",
      "worktree": "AutoGPT6",
      "worktree_path": "/Users/majdyz/Code/AutoGPT6",
      "branch": "feat/my-feature",
      "objective": "Implement X and open a PR",
      "state": "running",
      "last_output_hash": "abc123",
      "last_seen_at": 1712345678,
      "idle_since": 0,
      "revision_count": 0
    }
  ]
}
```

Agent states: `running` | `idle` | `stuck` | `waiting_approval` | `complete` | `done` | `escalated`

## Scripts

```
SKILLS_DIR=$(git rev-parse --show-toplevel)/.claude/skills/orchestrate
POLL_SCRIPT=$SKILLS_DIR/scripts/poll-cycle.sh
CLASSIFY_SCRIPT=$SKILLS_DIR/scripts/classify-pane.sh
STATE_FILE=~/.claude/orchestrator-state.json
```

## Subcommand: start

Collect fleet info from the user, write the state file, and start the CronCreate polling loop.

### Step 1 — Gather fleet info

Ask the user:
1. tmux session name (default: `work`)
2. For each agent window: `window` (e.g. `work:0`), `worktree` name (e.g. `AutoGPT6`), `objective` (what it's supposed to do)
3. Optional: `idle_threshold_seconds` (default: 300)

### Step 2 — Write state file

```bash
cat > ~/.claude/orchestrator-state.json << 'EOF'
{
  "active": true,
  "tmux_session": "TMUX_SESSION",
  "idle_threshold_seconds": THRESHOLD,
  "cron_job_id": null,
  "last_poll_at": 0,
  "agents": [AGENTS_ARRAY]
}
EOF
```

### Step 3 — Verify tmux connectivity

For each registered agent, verify the window exists and capture its current state:

```bash
for WINDOW in WINDOW_LIST; do
  tmux list-windows -t "${WINDOW%%:*}" 2>/dev/null && echo "$WINDOW: OK" || echo "$WINDOW: NOT FOUND"
  tmux display-message -t "$WINDOW" -p '#{pane_current_command}' 2>/dev/null
done
```

### Step 4 — Run first poll manually

```bash
SKILLS_DIR=$(git rev-parse --show-toplevel)/.claude/skills/orchestrate
bash $SKILLS_DIR/scripts/poll-cycle.sh
```

Read the output and process any immediate actions (see **Subcommand: poll** below).

### Step 5 — Start CronCreate loop

Use CronCreate to fire the poll cycle every 2 minutes. The poll prompt must be self-contained:

The cron prompt to use (fill in SKILLS_DIR with the actual path):
```
Orchestrator poll cycle — run: bash SKILLS_DIR/scripts/poll-cycle.sh

Read its JSON output array. For each element:
- action "kick": run tmux send-keys -t <window> "Continue with your current task. If all work is done output ORCHESTRATOR:DONE" Enter
- action "approve": run tmux send-keys -t <window> "y" Enter  
- action "complete": output "AGENT COMPLETE: <window> (<worktree>)" so the user sees it

Then output a 1-line summary: "Poll done — N agents: X running, Y idle/kicked, Z complete"
```

After CronCreate returns a job ID, store it in the state file:

```bash
jq --arg id "CRON_JOB_ID" '.cron_job_id = $id' ~/.claude/orchestrator-state.json > /tmp/orch.tmp \
  && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

Tell the user:
- The loop is running — it fires every ~2 minutes while this Claude session is open
- **CronCreate is session-scoped**: the loop stops when this Claude session closes
- To stop monitoring: run `/orchestrate stop`
- The loop auto-expires after 7 days per CronCreate limits

## Subcommand: stop

```bash
# Mark inactive
jq '.active = false' ~/.claude/orchestrator-state.json > /tmp/orch.tmp \
  && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json

# Show current state before stopping
jq '.agents[] | {window, state, worktree}' ~/.claude/orchestrator-state.json
```

Then use CronDelete with the stored job ID to cancel the loop:
```bash
CRON_ID=$(jq -r '.cron_job_id // ""' ~/.claude/orchestrator-state.json)
```

If `CRON_ID` is empty or null, the job was already gone (session ended). Tell the user.

## Subcommand: status

Display a readable summary of the fleet:

```bash
cat ~/.claude/orchestrator-state.json | jq -r '
  "=== Orchestrator Status ===",
  "Active: \(.active)",
  "Session: \(.tmux_session)",
  "Last poll: \(if .last_poll_at == 0 then "never" else (.last_poll_at | strftime("%H:%M:%S")) end)",
  "Idle threshold: \(.idle_threshold_seconds)s",
  "",
  "Agents:",
  (.agents[] | "  [\(.state | ascii_upcase)] \(.window) → \(.worktree): \(.objective | .[0:60])")
'
```

Also show pane command for each agent:

```bash
for WINDOW in $(jq -r '.agents[].window' ~/.claude/orchestrator-state.json); do
  CMD=$(tmux display-message -t "$WINDOW" -p '#{pane_current_command}' 2>/dev/null || echo "unreachable")
  echo "  $WINDOW foreground: $CMD"
done
```

## Subcommand: add

Add a new agent to an already-running orchestrator without restarting.

Ask the user for: `window`, `worktree`, `worktree_path`, `branch`, `objective`.

```bash
NEW_AGENT=$(jq -n \
  --arg window "WINDOW" \
  --arg worktree "WORKTREE" \
  --arg path "WORKTREE_PATH" \
  --arg branch "BRANCH" \
  --arg obj "OBJECTIVE" \
  '{window:$window, worktree:$worktree, worktree_path:$path, branch:$branch,
    objective:$obj, state:"running", last_output_hash:"", last_seen_at:0,
    idle_since:0, revision_count:0}')

jq --argjson a "$NEW_AGENT" '.agents += [$a]' ~/.claude/orchestrator-state.json > /tmp/orch.tmp \
  && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

Immediately run a classify on the new window to get its current state:

```bash
SKILLS_DIR=$(git rev-parse --show-toplevel)/.claude/skills/orchestrate
bash $SKILLS_DIR/scripts/classify-pane.sh WINDOW
```

## Subcommand: poll

Run one poll cycle manually (also called by the CronCreate loop).

```bash
SKILLS_DIR=$(git rev-parse --show-toplevel)/.claude/skills/orchestrate
ACTIONS=$(bash $SKILLS_DIR/scripts/poll-cycle.sh)
echo "Actions: $ACTIONS"
```

Read the JSON array output. For each action object, take the appropriate action:

### action: "kick" (idle or stuck agent)

The agent process has exited (shell is foreground) or its output has been frozen for too long.

```bash
# Re-enter the agent's worktree and restart claude
tmux send-keys -t WINDOW "cd WORKTREE_PATH && claude --permission-mode bypassPermissions" Enter
```

Wait 3 seconds for the shell to process, then send the continuation message:
```bash
sleep 3
tmux send-keys -t WINDOW "Continue with your current task: OBJECTIVE. When done, output ORCHESTRATOR:DONE" Enter
```

For **stuck** agents (claude is still running but output frozen):
```bash
# Send a nudge — do NOT restart, just prompt
tmux send-keys -t WINDOW "Continue with your current task. Review any errors and proceed. When done output ORCHESTRATOR:DONE" Enter
```

Increment the agent's `revision_count` in the state file. If `revision_count >= 3`, mark as `escalated` instead and notify the user that this agent needs human attention.

### action: "approve" (waiting for approval)

The agent has a confirmation prompt. Classify the pending command first:

```bash
tmux capture-pane -t WINDOW -p | tail -20
```

**Always send `y` for:**
- `git add`, `git commit`, `git push` (to non-main/master branches)
- `git checkout`, `git branch`, `git merge`
- `mkdir`, `cp`, `mv` within worktree path
- `npm install`, `pnpm install`, `poetry install`, `pip install`
- `docker compose up/down/build/stop`
- `pytest`, `pnpm test`, `cargo test`, `go test`
- `curl` to `localhost` only

**Escalate (mark as `escalated`, alert user) for:**
- `rm -rf` with paths outside the worktree
- `git push --force` or `git push -f`
- `git reset --hard`
- `sudo` any command
- Any command touching `/etc/`, `/usr/`, `/sys/`
- Writing API keys or secrets to files
- `DROP TABLE`, `DELETE FROM` without `WHERE`

```bash
# Safe: approve
tmux send-keys -t WINDOW "y" Enter

# Unsafe: DO NOT approve — escalate
jq --arg w "WINDOW" '
  .agents[] |= if .window == $w then .state = "escalated" else . end
' ~/.claude/orchestrator-state.json > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

Output a clear warning to the user about the escalated approval.

### action: "complete"

The agent output `ORCHESTRATOR:DONE`. Mark it done in the state file:

```bash
jq --arg w "WINDOW" '
  .agents[] |= if .window == $w then .state = "done" else . end
' ~/.claude/orchestrator-state.json > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

Output to the user: `AGENT DONE: WINDOW (WORKTREE) — OBJECTIVE`

Optionally, ask the user if they want to run `/pr-test WORKTREE_PATH` on the completed agent.

## Poll cycle summary

After processing all actions, output a one-line summary:

```
Poll [HH:MM:SS] — N agents | X running | Y kicked | Z approved | W complete | V escalated
```

## Key rules

1. **Never restart an agent that is still running** — only restart when the foreground process is a shell
2. **bypassPermissions is the default** — agents launched with it should rarely need approval. If you see frequent approval prompts, something is wrong — escalate rather than blindly approving
3. **Increment revision_count on every kick** — after 3 kicks to the same agent, escalate to user
4. **Atomic state writes** — always write to `.tmp` then `mv` to avoid corruption
5. **Never approve destructive commands** — when in doubt, escalate
6. **The loop is session-scoped** — remind the user of this when they run `start`
