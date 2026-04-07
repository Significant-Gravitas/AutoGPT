---
name: orchestrate
description: "Meta-agent supervisor that manages a fleet of Claude Code agents running in tmux windows. Auto-discovers spare worktrees, spawns agents, monitors state, kicks idle agents, approves safe confirmations, and recycles worktrees when done. TRIGGER when user asks to supervise agents, run parallel tasks, manage worktrees, check agent status, or orchestrate parallel work."
user-invocable: true
argument-hint: "any free text — e.g. 'start 3 agents on X Y Z', 'show status', 'add task: implement feature A', 'stop', 'how many are free?'"
metadata:
  author: autogpt-team
  version: "5.0.0"
---

# Orchestrate — Agent Fleet Supervisor

One tmux session, N windows — each window is one agent working in its own worktree. Speak naturally; Claude maps your intent to the right scripts.

## Scripts

```bash
SKILLS_DIR=$(git rev-parse --show-toplevel)/.claude/skills/orchestrate/scripts
STATE_FILE=~/.claude/orchestrator-state.json
```

| Script | Purpose |
|---|---|
| `find-spare.sh [REPO_ROOT]` | List free worktrees — one `PATH BRANCH` per line |
| `spawn-agent.sh SESSION PATH SPARE NEW_BRANCH OBJECTIVE [PR_NUMBER] [STEPS...]` | Create window + checkout branch + launch claude + send task. **Stdout: `SESSION:WIN` only** |
| `recycle-agent.sh WINDOW PATH SPARE_BRANCH` | Kill window + restore spare branch |
| `run-loop.sh` | **Mechanical babysitter** — idle restart + dialog approval + recycle on ORCHESTRATOR:DONE + supervisor health check + all-done notification |
| `verify-complete.sh WINDOW` | Verify PR is done: checkpoints ✓ + 0 unresolved threads + CI green. Repo auto-derived from state file `.repo` or git remote. |
| `notify.sh MESSAGE` | Send notification via Discord webhook (env `DISCORD_WEBHOOK_URL` or state `.discord_webhook`), macOS notification center, and stdout |
| `capacity.sh [REPO_ROOT]` | Print available + in-use worktrees |
| `status.sh` | Print fleet status + live pane commands |
| `poll-cycle.sh` | One monitoring cycle — classifies panes, tracks checkpoints, returns JSON action array |
| `classify-pane.sh WINDOW` | Classify one pane state |

## Two-layer supervision model

```
Supervisor window (dedicated Claude Code in tmux, runs continuously)
  └── Reads pane output, checks CI, intervenes with targeted guidance
        run-loop.sh (separate tmux window, every 30s)
          └── Mechanical only: idle restart, dialog approval, recycle on ORCHESTRATOR:DONE
```

**Supervisor window** is the intelligence layer — a dedicated Claude Code instance running in its own tmux window. It reads each agent's pane, checks PR CI/threads, and sends specific guidance when agents stall, deviate, or claim completion without meeting all criteria. Running in its own window means it doesn't pollute the user's main Claude session context, and survives context compression.

**run-loop.sh** is the mechanical layer — zero tokens, handles things that need no judgment: restart crashed agents, press Enter on dialogs, recycle completed worktrees (only after `verify-complete.sh` passes).

## Checkpoint protocol

Agents output checkpoints as they complete each required step:

```
CHECKPOINT:<step-name>
```

Required steps are passed as args to `spawn-agent.sh` (e.g. `pr-address pr-test`). `run-loop.sh` will not recycle a window until all required checkpoints are found in the pane output. If `verify-complete.sh` fails, the agent is re-briefed automatically.

## Worktree lifecycle

```text
spare/N branch  →  spawn-agent.sh  →  window + feat/branch + claude running
                                                ↓
                                  CHECKPOINT:<step> (as steps complete)
                                                ↓
                                         ORCHESTRATOR:DONE
                                                ↓
                          verify-complete.sh: checkpoints ✓ + 0 threads + CI green
                                                ↓
                               recycle-agent.sh → spare/N (free again)
```

## State file (`~/.claude/orchestrator-state.json`)

Never committed to git. You maintain this file directly using `jq` + atomic writes (`.tmp` → `mv`).

```json
{
  "active": true,
  "tmux_session": "autogpt1",
  "idle_threshold_seconds": 300,
  "loop_window": "autogpt1:5",
  "supervisor_window": "autogpt1:6",
  "repo": "Significant-Gravitas/AutoGPT",
  "discord_webhook": "https://discord.com/api/webhooks/...",
  "last_poll_at": 0,
  "agents": [
    {
      "window": "autogpt1:3",
      "worktree": "AutoGPT6",
      "worktree_path": "/path/to/AutoGPT6",
      "spare_branch": "spare/6",
      "branch": "feat/my-feature",
      "objective": "Implement X and open a PR",
      "pr_number": "12345",
      "steps": ["pr-address", "pr-test"],
      "checkpoints": ["pr-address"],
      "state": "running",
      "last_output_hash": "",
      "last_seen_at": 0,
      "idle_since": 0,
      "revision_count": 0,
      "last_rebriefed_at": 0
    }
  ]
}
```

Top-level optional fields:
- `repo` — GitHub `owner/repo` for CI/thread checks. Auto-derived from git remote if omitted.
- `discord_webhook` — Discord webhook URL for completion notifications. Also reads `DISCORD_WEBHOOK_URL` env var.
- `supervisor_window` — tmux window running the supervisor Claude; `run-loop.sh` restarts it if it exits.

Per-agent optional fields:
- `last_rebriefed_at` — Unix timestamp of last re-brief; enforces 5-min cooldown to prevent spam.

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

Use an existing session. **Never create a tmux session from within Claude** — it becomes a child of Claude's process and dies when the session ends. If no session exists, tell the user to run `tmux new-session -d -s autogpt1` in their terminal first, then re-invoke `/orchestrate`.

### 2. Show available capacity

```bash
bash $SKILLS_DIR/capacity.sh $(git rev-parse --show-toplevel)
```

### 3. Collect tasks from the user

For each task, gather:
- **objective** — what to do (e.g. "implement feature X and open a PR")
- **branch name** — e.g. `feat/my-feature` (derive from objective if not given)
- **pr_number** — GitHub PR number if working on an existing PR (for verification)
- **steps** — required checkpoint names in order (e.g. `pr-address pr-test`) — derive from objective

Ask for `idle_threshold_seconds` only if the user mentions it (default: 300).

Never ask the user to specify a worktree — auto-assign from `find-spare.sh`.

### 4. Spawn one agent per task

```bash
# Get ordered list of spare worktrees
SPARE_LIST=$(bash $SKILLS_DIR/find-spare.sh $(git rev-parse --show-toplevel))

# For each task, take the next spare line:
WORKTREE_PATH=$(echo "$SPARE_LINE" | awk '{print $1}')
SPARE_BRANCH=$(echo "$SPARE_LINE" | awk '{print $2}')

# With PR number and required steps:
WINDOW=$(bash $SKILLS_DIR/spawn-agent.sh "$SESSION" "$WORKTREE_PATH" "$SPARE_BRANCH" "$NEW_BRANCH" "$OBJECTIVE" "$PR_NUMBER" "pr-address" "pr-test")

# Without PR (new work):
WINDOW=$(bash $SKILLS_DIR/spawn-agent.sh "$SESSION" "$WORKTREE_PATH" "$SPARE_BRANCH" "$NEW_BRANCH" "$OBJECTIVE")
```

Build an agent record and append it to the state file. If the state file doesn't exist yet, initialize it:

```bash
# Derive repo from git remote (used by verify-complete.sh + supervisor)
REPO=$(git remote get-url origin 2>/dev/null | sed 's|.*github\.com[:/]||; s|\.git$||' || echo "")

jq -n \
  --arg session "$SESSION" \
  --arg repo "$REPO" \
  --argjson threshold 300 \
  '{active:true, tmux_session:$session, idle_threshold_seconds:$threshold,
    repo:$repo, loop_window:null, supervisor_window:null, last_poll_at:0, agents:[]}' \
  > ~/.claude/orchestrator-state.json
```

Optionally add a Discord webhook for completion notifications:
```bash
jq --arg hook "$DISCORD_WEBHOOK_URL" '.discord_webhook = $hook' ~/.claude/orchestrator-state.json \
  > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

Append each new agent:
```bash
jq --argjson a "$NEW_AGENT" '.agents += [$a]' ~/.claude/orchestrator-state.json \
  > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

### 5. Start both supervision layers

**Layer 1 — mechanical babysitter** (tmux window, zero tokens):
```bash
LOOP_WIN=$(tmux new-window -t "$SESSION" -n "orchestrator" -P -F '#{window_index}')
LOOP_WINDOW="${SESSION}:${LOOP_WIN}"
tmux send-keys -t "$LOOP_WINDOW" "bash $SKILLS_DIR/run-loop.sh" Enter

jq --arg w "$LOOP_WINDOW" '.loop_window = $w' ~/.claude/orchestrator-state.json \
  > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

**Layer 2 — intelligent supervisor** (dedicated Claude Code window):

Open a new tmux window in the SAME session and launch Claude there:

```bash
SUP_WIN=$(tmux new-window -t "$SESSION" -n "supervisor" -P -F '#{window_index}')
SUP_WINDOW="${SESSION}:${SUP_WIN}"
SKILLS_DIR_ABS="$(git -C "$(git rev-parse --show-toplevel)" rev-parse --show-toplevel)/.claude/skills/orchestrate/scripts"

tmux send-keys -t "$SUP_WINDOW" "cd '$(git rev-parse --show-toplevel)' && claude --permission-mode bypassPermissions" Enter
# Wait for claude to be ready (same pattern as spawn-agent.sh)
# Then send the supervisor prompt:
```

Send this prompt to the supervisor window (fill in real session, window IDs, and skills dir):

```text
You are the intelligent supervisor for a Claude agent fleet.

Your job: every 2-3 minutes, check all running agents and intervene when needed.

SELF-RECOVERY: If your own context compacts and you lose track of what you were doing,
re-read the state file: cat ~/.claude/orchestrator-state.json | jq
That gives you the full picture. Resume supervision immediately.

State file: cat ~/.claude/orchestrator-state.json | jq

SKILLS_DIR: <SKILLS_DIR_ABS>

For each running agent, read their pane:
  tmux capture-pane -t SESSION:WIN -p -S -200 | tail -80

For each agent decide:
- Actively working (spinner/tools running) → do nothing
- Idle at ❯ prompt without ORCHESTRATOR:DONE → stalled; send specific nudge:
    tmux send-keys -t SESSION:WIN "Your objective: <restate from state file>. Continue from where you left off." Enter
- Stuck in loop / repeating error → send targeted fix guidance
- Waiting for input / asking a question → answer and unblock
- CI red → run: gh pr checks PR_NUMBER --repo $(jq -r '.repo // "Significant-Gravitas/AutoGPT"' ~/.claude/orchestrator-state.json)
    then tell the agent exactly what's failing and how to fix it
- Context compacted, agent appears lost → send recovery command:
    tmux send-keys -t SESSION:WIN "Run: cat ~/.claude/orchestrator-state.json | jq '.agents[] | select(.window==\"SESSION:WIN\")' and gh pr view PR_NUMBER --json title,body,headRefName to reorient, then continue your task." Enter
- Claims ORCHESTRATOR:DONE → run verify-complete.sh:
    bash SKILLS_DIR/verify-complete.sh SESSION:WIN
    If it fails, re-brief the agent with the specific failure reason.

Only surface to the user if a strategic decision is needed.
When all agents reach state "done" or "escalated", your job is complete.

Begin your first check now, then loop every 2-3 minutes.
```

Store the supervisor window in the state file:

```bash
jq --arg w "$SUP_WINDOW" '.supervisor_window = $w' ~/.claude/orchestrator-state.json \
  > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

## Adding an agent

Find the next spare worktree, then spawn and append to state — same as steps 2–4 above but for a single task. If no spare worktrees are available, tell the user.

## Supervisor duties (what the supervisor window does every 2-3 min)

1. Read `~/.claude/orchestrator-state.json` to get agent windows and objectives
2. For each agent in `running`/`idle`/`stuck` state, capture pane output
3. **Check for stalling**: no new output in last 5+ min → send specific re-orientation message
4. **Check for deviation**: agent doing something unrelated → correct it
5. **Check for PR issues**: unresolved threads, CI failures → inform agent with specifics
6. **Verify claims of completion**: run `verify-complete.sh` when agent says ORCHESTRATOR:DONE (run-loop.sh also does this, but you catch it faster)
7. **Re-brief after context compaction**: if agent asks "what should I do?" or appears lost, run:
   ```bash
   cat ~/.claude/orchestrator-state.json | jq '.agents[] | select(.window=="SESSION:WIN")'
   gh pr view PR_NUMBER --json title,body,headRefName
   ```
   Then send the agent its objective + PR context via tmux

## Self-recovery protocol (agents)

spawn-agent.sh automatically includes this instruction in every objective:

> If your context compacts and you lose track of what to do, run:
> `cat ~/.claude/orchestrator-state.json | jq '.agents[] | select(.window=="SESSION:WIN")'`
> and `gh pr view PR_NUMBER --json title,body,headRefName` to reorient.
> Output each completed step as `CHECKPOINT:<step-name>` on its own line.

## Stopping

```bash
# Mark inactive — run-loop.sh checks this and exits cleanly
jq '.active = false' ~/.claude/orchestrator-state.json > /tmp/orch.tmp \
  && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json

# Kill the orchestrator window
LOOP_WINDOW=$(jq -r '.loop_window // ""' ~/.claude/orchestrator-state.json)
[ -n "$LOOP_WINDOW" ] && tmux kill-window -t "$LOOP_WINDOW" 2>/dev/null || true
```

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
9. **Never recycle without verification** — `verify-complete.sh` must pass before recycling
10. **No TASK.md files** — commit risk; use state file + `gh pr view` for agent context persistence
11. **Re-brief stalled agents** — read objective from state file + `gh pr view`, send via tmux
