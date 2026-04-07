---
name: orchestrate
description: "Meta-agent supervisor that manages a fleet of Claude Code agents running in tmux windows. Auto-discovers spare worktrees, spawns agents, monitors state, kicks idle agents, approves safe confirmations, and recycles worktrees when done. TRIGGER when user asks to supervise agents, run parallel tasks, manage worktrees, check agent status, or orchestrate parallel work."
user-invocable: true
argument-hint: "any free text — e.g. 'start 3 agents on X Y Z', 'show status', 'add task: implement feature A', 'stop', 'how many are free?'"
metadata:
  author: autogpt-team
  version: "3.0.0"
---

# Orchestrate — Agent Fleet Supervisor

One tmux session, N windows — each window is one agent working in its own worktree. Speak naturally; Claude maps your intent to the right scripts.

## Scripts

```bash
SKILLS_DIR=$(git rev-parse --show-toplevel)/.claude/skills/orchestrate/scripts
STATE_FILE=~/.claude/orchestrator-state.json
```

| Script | Purpose | Key args |
|---|---|---|
| `find-spare.sh [REPO_ROOT]` | List free worktrees — one `PATH BRANCH` per line | |
| `spawn-agent.sh SESSION PATH SPARE NEW_BRANCH OBJECTIVE` | Create window + checkout branch + launch claude + send task. **Stdout: `SESSION:WIN` only** | |
| `recycle-agent.sh WINDOW PATH SPARE_BRANCH` | Kill window + restore spare branch | |
| `capacity.sh [REPO_ROOT]` | Print available + in-use worktrees | |
| `status.sh` | Print fleet status + live pane commands | |
| `poll-cycle.sh` | One monitoring cycle — returns JSON action array | |
| `classify-pane.sh WINDOW` | Classify one pane state | |

## Worktree lifecycle

```text
spare/N branch  →  spawn-agent.sh  →  window + feat/branch + claude running
                                                ↓
                                         ORCHESTRATOR:DONE
                                                ↓
                               recycle-agent.sh → spare/N (free again)
```

## State file (`~/.claude/orchestrator-state.json`)

Never committed to git. Claude maintains this file directly using `jq` + atomic writes (`.tmp` → `mv`).

```json
{
  "active": true,
  "tmux_session": "autogpt1",
  "idle_threshold_seconds": 300,
  "cron_job_id": "...",
  "last_poll_at": 0,
  "agents": [
    {
      "window": "autogpt1:3",
      "worktree": "AutoGPT6",
      "worktree_path": "/path/to/AutoGPT6",
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

## Intent → action mapping

Match the user's message to one of these intents:

| The user says something like… | What to do |
|---|---|
| "status", "what's running", "show agents" | Run `status.sh` + `capacity.sh`, show output |
| "how many free", "capacity", "available worktrees" | Run `capacity.sh`, show output |
| "start N agents on X, Y, Z" or "run these tasks: …" | See **Spawning agents** below |
| "add task: …", "add one more agent for …" | See **Adding an agent** below |
| "stop", "shut down", "pause the fleet" | See **Stopping** below |
| "poll", "check now", "run a cycle" | Run `poll-cycle.sh`, process actions |
| "recycle window X", "free up autogpt3" | Run `recycle-agent.sh` directly |

When the intent is ambiguous, show capacity first and ask what tasks to run.

## Spawning agents

### 1. Resolve tmux session

```bash
tmux list-sessions -F "#{session_name}: #{session_windows} windows" 2>/dev/null
```

Use the existing session. If none exist, create one:
```bash
tmux new-session -d -s autogpt1
SESSION="autogpt1"
```

### 2. Show available capacity

```bash
bash $SKILLS_DIR/capacity.sh $(git rev-parse --show-toplevel)
```

### 3. Collect tasks from the user

For each task, gather:
- **objective** — what to do (e.g. "implement feature X and open a PR")
- **branch name** — e.g. `feat/my-feature` (derive from objective if not given)

Ask for `idle_threshold_seconds` only if the user mentions it (default: 300).

Never ask the user to specify a worktree — auto-assign from `find-spare.sh`.

### 4. Spawn one agent per task

```bash
# Get ordered list of spare worktrees
SPARE_LIST=$(bash $SKILLS_DIR/find-spare.sh $(git rev-parse --show-toplevel))

# For each task, take the next spare line:
WORKTREE_PATH=$(echo "$SPARE_LINE" | awk '{print $1}')
SPARE_BRANCH=$(echo "$SPARE_LINE" | awk '{print $2}')

WINDOW=$(bash $SKILLS_DIR/spawn-agent.sh "$SESSION" "$WORKTREE_PATH" "$SPARE_BRANCH" "$NEW_BRANCH" "$OBJECTIVE")
```

Build an agent record and append it to the state file. If the state file doesn't exist yet, initialize it:

```bash
jq -n \
  --arg session "$SESSION" \
  --argjson threshold 300 \
  '{active:true, tmux_session:$session, idle_threshold_seconds:$threshold,
    cron_job_id:null, last_poll_at:0, agents:[]}' \
  > ~/.claude/orchestrator-state.json
```

Append each new agent:
```bash
jq --argjson a "$NEW_AGENT" '.agents += [$a]' ~/.claude/orchestrator-state.json \
  > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

### 5. Run first poll, then start the cron

```bash
bash $SKILLS_DIR/poll-cycle.sh | jq .
```

Then start CronCreate with this prompt (substitute the real absolute path for `SKILLS_DIR`):

```text
Orchestrator poll cycle.

Run: bash SKILLS_DIR/poll-cycle.sh

Read the JSON output array. For each action object:

- action=="kick" AND state=="idle" (agent exited — shell is foreground):
  Restart the agent:
    tmux send-keys -t <window> "cd '<worktree_path>' && claude --permission-mode bypassPermissions" Enter
  Wait 3s. Capture pane: if "Enter to confirm" visible, send Down+Enter first.
  Then send: "<objective>. When all work is done, output the exact string: ORCHESTRATOR:DONE" Enter

- action=="kick" AND state=="stuck" (claude still running but frozen):
  Nudge only — do NOT restart:
    tmux send-keys -t <window> "Continue with your task. Review any errors and proceed. When done output ORCHESTRATOR:DONE" Enter

- action=="approve":
  Capture pane: tmux capture-pane -t <window> -p | tail -5
  SAFE to approve: git operations, install/build, tests, docker, curl to localhost
  ESCALATE (set state=escalated, do not send keys) for: rm -rf outside worktree, force push to main/master, sudo, writing secrets to files
  Settings dialog "Enter to confirm": send Down+Enter

- action=="complete":
  Recycle the worktree:
    bash SKILLS_DIR/recycle-agent.sh <window> <worktree_path> <spare_branch>
  Update state file: set agent .state = "done"
    jq --arg w "<window>" '.agents |= map(if .window == $w then .state = "done" else . end)' \
      ~/.claude/orchestrator-state.json > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
  Print: "AGENT DONE + RECYCLED: <window> (<worktree>) → <spare_branch>"

After processing all actions, print one summary line:
"Poll [HH:MM] — N agents: X running, Y kicked, Z done+recycled, V escalated"
```

Store the returned cron job ID: update `.cron_job_id` in the state file.

## Adding an agent

Find the next spare worktree, then spawn and append to state — same as steps 2–4 above but for a single task. If no spare worktrees are available, tell the user.

## Stopping

```bash
# Mark inactive so poll-cycle.sh exits early (active==false)
jq '.active = false' ~/.claude/orchestrator-state.json > /tmp/orch.tmp \
  && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

Cancel the cron job using CronDelete with `.cron_job_id` from the state file.

Does **not** recycle running worktrees — agents may still be mid-task. Run `capacity.sh` to see what's still in progress.

## Key rules

1. **Scripts do all the heavy lifting** — don't reimplement their logic inline in this file
2. **Never ask the user to pick a worktree** — auto-assign from `find-spare.sh` output
3. **Never restart a running agent** — only restart on `idle` kicks (foreground is a shell)
4. **Auto-dismiss settings dialogs** — if "Enter to confirm" appears, send Down+Enter
5. **Always `--permission-mode bypassPermissions`** on every spawn
6. **Escalate after 3 kicks** — mark `escalated`, surface to user
7. **Atomic state writes** — always write to `.tmp` then `mv`
8. **Never approve destructive commands** outside the worktree scope — when in doubt, escalate
