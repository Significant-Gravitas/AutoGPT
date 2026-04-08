---
name: orchestrate
description: "Meta-agent supervisor that manages a fleet of Claude Code agents running in tmux windows. Auto-discovers spare worktrees, spawns agents, monitors state, kicks idle agents, approves safe confirmations, and recycles worktrees when done. TRIGGER when user asks to supervise agents, run parallel tasks, manage worktrees, check agent status, or orchestrate parallel work."
user-invocable: true
argument-hint: "any free text — e.g. 'start 3 agents on X Y Z', 'show status', 'add task: implement feature A', 'stop', 'how many are free?'"
metadata:
  author: autogpt-team
  version: "6.0.0"
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
| `verify-complete.sh WINDOW` | Verify PR is done: checkpoints ✓ + 0 unresolved threads + CI green + no fresh CHANGES_REQUESTED. Repo auto-derived from state file `.repo` or git remote. |
| `notify.sh MESSAGE` | Send notification via Discord webhook (env `DISCORD_WEBHOOK_URL` or state `.discord_webhook`), macOS notification center, and stdout |
| `capacity.sh [REPO_ROOT]` | Print available + in-use worktrees |
| `status.sh` | Print fleet status + live pane commands |
| `poll-cycle.sh` | One monitoring cycle — classifies panes, tracks checkpoints, returns JSON action array |
| `classify-pane.sh WINDOW` | Classify one pane state |

## Supervision model

```
Orchestrating Claude (this Claude session — IS the supervisor)
  └── Reads pane output, checks CI, intervenes with targeted guidance
        run-loop.sh (separate tmux window, every 30s)
          └── Mechanical only: idle restart, dialog approval, recycle on ORCHESTRATOR:DONE
```

**You (the orchestrating Claude)** are the supervisor. After spawning agents, stay in this conversation and actively monitor: poll each agent's pane every 2-3 minutes, check CI, nudge stalled agents, and verify completions. Do not spawn a separate supervisor Claude window — it loses context, is hard to observe, and compounds context compression problems.

**run-loop.sh** is the mechanical layer — zero tokens, handles things that need no judgment: restart crashed agents, press Enter on dialogs, recycle completed worktrees (only after `verify-complete.sh` passes).

## Checkpoint protocol

Agents output checkpoints as they complete each required step:

```
CHECKPOINT:<step-name>
```

Required steps are passed as args to `spawn-agent.sh` (e.g. `pr-address pr-test`). `run-loop.sh` will not recycle a window until all required checkpoints are found in the pane output. If `verify-complete.sh` fails, the agent is re-briefed automatically.

## Worktree lifecycle

```text
spare/N branch  →  spawn-agent.sh (--session-id UUID)  →  window + feat/branch + claude running
                                                                 ↓
                                               CHECKPOINT:<step> (as steps complete)
                                                                 ↓
                                                        ORCHESTRATOR:DONE
                                                                 ↓
                                    verify-complete.sh: checkpoints ✓ + 0 threads + CI green + no fresh CHANGES_REQUESTED
                                                                 ↓
                                              state → "done", notify, window KEPT OPEN
                                                                 ↓
                              user/orchestrator explicitly requests recycle
                                                                 ↓
                                         recycle-agent.sh → spare/N (free again)
```

**Windows are never auto-killed.** The worktree stays on its branch, the session stays alive. The agent is done working but the window, git state, and Claude session are all preserved until you choose to recycle.

**To resume a done or crashed session:**
```bash
# Resume by stored session ID (preferred — exact session, full context)
claude --resume SESSION_ID --permission-mode bypassPermissions

# Or resume most recent session in that worktree directory
cd /path/to/worktree && claude --continue --permission-mode bypassPermissions
```

**To manually recycle when ready:**
```bash
bash ~/.claude/orchestrator/scripts/recycle-agent.sh SESSION:WIN WORKTREE_PATH spare/N
# Then update state:
jq --arg w "SESSION:WIN" '.agents |= map(if .window == $w then .state = "recycled" else . end)' \
  ~/.claude/orchestrator-state.json > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

## State file (`~/.claude/orchestrator-state.json`)

Never committed to git. You maintain this file directly using `jq` + atomic writes (`.tmp` → `mv`).

```json
{
  "active": true,
  "tmux_session": "autogpt1",
  "idle_threshold_seconds": 300,
  "loop_window": "autogpt1:5",
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
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "steps": ["pr-address", "pr-test"],
      "checkpoints": ["pr-address"],
      "state": "running",
      "last_output_hash": "",
      "last_seen_at": 0,
      "spawned_at": 0,
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

Per-agent fields:
- `session_id` — UUID passed to `claude --session-id` at spawn; use with `claude --resume UUID` to restore exact session context after a crash or window close.
- `last_rebriefed_at` — Unix timestamp of last re-brief; enforces 5-min cooldown to prevent spam.

Agent states: `running` | `idle` | `stuck` | `waiting_approval` | `complete` | `done` | `escalated`

`done` means verified complete — window is still open, session still alive, worktree still on task branch. Not recycled yet.

## Serial /pr-test rule

`/pr-test` and `/pr-test --fix` run local Docker + integration tests that use shared ports, a shared database, and shared build caches. **Running two `/pr-test` jobs simultaneously will cause port conflicts and database corruption.**

**Rule: only one `/pr-test` runs at a time. The orchestrator serializes them.**

You (the orchestrating Claude) own the test queue:
1. Agents do `pr-review` and `pr-address` in parallel — that's safe (they only push code and reply to GitHub).
2. When a PR needs local testing, add it to your mental queue — don't give agents a `pr-test` step.
3. Run `/pr-test https://github.com/OWNER/REPO/pull/PR_NUMBER --fix` yourself, sequentially.
4. Feed results back to the relevant agent via `tmux send-keys`:
   ```bash
   tmux send-keys -t SESSION:WIN "Local tests for PR #N: <paste failure output or 'all passed'>. Fix any failures and push, then output ORCHESTRATOR:DONE."
   sleep 0.3
   tmux send-keys -t SESSION:WIN Enter
   ```
5. Wait for CI to confirm green before marking the agent done.

If multiple PRs need testing at the same time, pick the one furthest along (fewest pending CI checks) and test it first. Only start the next test after the previous one completes.

## Session restore (tested and confirmed)

Agent sessions are saved to disk. To restore a closed or crashed session:

```bash
# If session_id is in state (preferred):
NEW_WIN=$(tmux new-window -t SESSION -n WORKTREE_NAME -P -F '#{window_index}')
tmux send-keys -t "SESSION:${NEW_WIN}" "cd /path/to/worktree && claude --resume SESSION_ID --permission-mode bypassPermissions" Enter

# If no session_id (use --continue for most recent session in that directory):
tmux send-keys -t "SESSION:${NEW_WIN}" "cd /path/to/worktree && claude --continue --permission-mode bypassPermissions" Enter
```

`--continue` restores the full conversation history including all tool calls, file edits, and context. The agent resumes exactly where it left off. After restoring, update the window address in the state file:

```bash
jq --arg old "SESSION:OLD_WIN" --arg new "SESSION:NEW_WIN" \
  '(.agents[] | select(.window == $old)).window = $new' \
  ~/.claude/orchestrator-state.json > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

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

`spawn-agent.sh` writes the initial agent record (window, worktree_path, branch, objective, state, etc.) to the state file automatically — **do not append the record again after calling it.** The record already exists and `pr_number`/`steps` are patched in by the script itself.

### 5. Start the mechanical babysitter

```bash
LOOP_WIN=$(tmux new-window -t "$SESSION" -n "orchestrator" -P -F '#{window_index}')
LOOP_WINDOW="${SESSION}:${LOOP_WIN}"
tmux send-keys -t "$LOOP_WINDOW" "bash $SKILLS_DIR/run-loop.sh" Enter

jq --arg w "$LOOP_WINDOW" '.loop_window = $w' ~/.claude/orchestrator-state.json \
  > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

### 6. Begin supervising directly in this conversation

You are the supervisor. After spawning, immediately start your first poll loop (see **Supervisor duties** below) and continue every 2-3 minutes. Do NOT spawn a separate supervisor Claude window.

## Adding an agent

Find the next spare worktree, then spawn and append to state — same as steps 2–4 above but for a single task. If no spare worktrees are available, tell the user.

## Supervisor duties (YOUR job, every 2-3 min in this conversation)

You are the supervisor. Run this poll loop directly in your Claude session — not in a separate window.

### Poll loop mechanism

You are reactive — you only act when a tool completes or the user sends a message. To create a self-sustaining poll loop without user involvement:

1. Start each poll with `run_in_background: true` + a sleep before the work:
   ```bash
   sleep 120 && tmux capture-pane -t autogpt1:0 -p -S -200 | tail -40
   # + similar for each active window
   ```
2. When the background job notifies you, read the pane output and take action.
3. Immediately schedule the next background poll — this keeps the loop alive.
4. Stop scheduling when all agents are done/escalated.

**Never tell the user "I'll poll every 2-3 minutes"** — that does nothing without a trigger. Start the background job instead.

### Each poll: what to check

```bash
# 1. Read state
cat ~/.claude/orchestrator-state.json | jq '.agents[] | {window, worktree, branch, state, pr_number, checkpoints}'

# 2. For each running/stuck/idle agent, capture pane
tmux capture-pane -t SESSION:WIN -p -S -200 | tail -60
```

For each agent, decide:

| What you see | Action |
|---|---|
| Spinner / tools running | Do nothing — agent is working |
| Idle `❯` prompt, no `ORCHESTRATOR:DONE` | Stalled — send specific nudge with objective from state |
| Stuck in error loop | Send targeted fix with exact error + solution |
| Waiting for input / question | Answer and unblock via `tmux send-keys` |
| CI red | `gh pr checks PR_NUMBER --repo REPO` → tell agent exactly what's failing |
| GitHub abuse rate limit error | Nudge: "Wait 60 seconds then continue posting replies with sleep 3 between each" |
| Context compacted / agent lost | Send recovery: `cat ~/.claude/orchestrator-state.json | jq '.agents[] | select(.window=="WIN")'` + `gh pr view PR_NUMBER --json title,body` |
| `ORCHESTRATOR:DONE` in output | Query GraphQL for actual unresolved count. If >0, re-brief. If 0, run `verify-complete.sh` |

**Poll all windows from state, not from memory.** Before each poll, run:
```bash
jq -r '.agents[] | select(.state | test("running|idle|stuck|waiting_approval|pending_evaluation")) | .window' ~/.claude/orchestrator-state.json
```
and capture every window listed. If you manually added a window outside spawn-agent.sh, ensure it's in the state file first.

### RUNNING count includes waiting_approval agents

The `RUNNING` count from run-loop.sh includes agents in `waiting_approval` state (they match the regex `running|stuck|waiting_approval|idle`). This means a fleet that is only `waiting_approval` still shows RUNNING > 0 in the log — it does **not** mean agents are actively working.

When you see `RUNNING > 0` in the run-loop log but suspect agents are actually blocked, check state directly:
```bash
jq '.agents[] | {window, state, worktree}' ~/.claude/orchestrator-state.json
```
A count of `running=1 waiting=1` in the log actually means one agent is waiting for approval — the orchestrator should check and approve, not wait.

### State file staleness recovery

The state file is written by scripts but can drift from reality when windows are closed, sessions expire, or the orchestrator restarts across conversations.

**Signs of stale state:**
- `loop_window` points to a window that no longer exists in the tmux session
- An agent's `state` is `running` but tmux window is closed or shows a shell prompt (not claude)
- `last_seen_at` is hours old but state still says `running`

**Recovery steps:**

1. **Verify actual tmux windows:**
```bash
tmux list-windows -t SESSION -F '#{window_index}: #{window_name} (#{pane_current_command})'
```

2. **Cross-reference with state file:**
```bash
jq -r '.agents[] | "\(.window) \(.state) \(.worktree)"' ~/.claude/orchestrator-state.json
```

3. **Fix stale entries:**
```bash
# Agent window closed — mark idle so run-loop.sh will restart it
jq --arg w "SESSION:WIN" '(.agents[] | select(.window==$w)).state = "idle"' \
  ~/.claude/orchestrator-state.json > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json

# loop_window gone — kill the stale reference, then restart run-loop.sh
jq '.loop_window = null' ~/.claude/orchestrator-state.json > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
LOOP_WIN=$(tmux new-window -t "$SESSION" -n "orchestrator" -P -F '#{window_index}')
LOOP_WINDOW="${SESSION}:${LOOP_WIN}"
tmux send-keys -t "$LOOP_WINDOW" "bash $SKILLS_DIR/run-loop.sh" Enter
jq --arg w "$LOOP_WINDOW" '.loop_window = $w' ~/.claude/orchestrator-state.json \
  > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

4. **After any state repair, re-run `status.sh` to confirm coherence before resuming supervision.**

### Strict ORCHESTRATOR:DONE gate

`verify-complete.sh` handles the main checks automatically (checkpoints, threads, CI green, spawned_at, and CHANGES_REQUESTED). Run it:

**CHANGES_REQUESTED staleness rule**: a `CHANGES_REQUESTED` review only blocks if it was submitted *after* the latest commit. If the latest commit postdates the review, the review is considered stale (feedback already addressed) and does not block. This avoids false negatives when a bot reviewer hasn't re-reviewed after the agent's fixing commits.

```bash
SKILLS_DIR=~/.claude/orchestrator/scripts
bash $SKILLS_DIR/verify-complete.sh SESSION:WIN
```

If it passes → run-loop.sh will recycle the window automatically. No manual action needed.
If it fails → re-brief the agent with the failure reason. Never manually mark state `done` to bypass this.

### Re-brief a stalled agent

**Before sending any nudge, verify the pane is at an idle ❯ prompt.** Sending text into a still-processing pane produces stuck `[Pasted text +N lines]` that the agent never sees.

Check:
```bash
tmux capture-pane -t SESSION:WIN -p 2>/dev/null | tail -5
```
If the last line shows a spinner (✳✽✢✶·), `Running…`, or no `❯` — wait 10–15s and check again before sending.

```bash
OBJ=$(jq -r --arg w SESSION:WIN '.agents[] | select(.window==$w) | .objective' ~/.claude/orchestrator-state.json)
PR=$(jq -r --arg w SESSION:WIN '.agents[] | select(.window==$w) | .pr_number' ~/.claude/orchestrator-state.json)
tmux send-keys -t SESSION:WIN "You appear stalled. Your objective: $OBJ. Check: gh pr view $PR --json title,body,headRefName to reorient."
sleep 0.3
tmux send-keys -t SESSION:WIN Enter
```

If `image_path` is set on the agent record, include: "Re-read context at IMAGE_PATH with the Read tool."

## Self-recovery protocol (agents)

spawn-agent.sh automatically includes this instruction in every objective:

> If your context compacts and you lose track of what to do, run:
> `cat ~/.claude/orchestrator-state.json | jq '.agents[] | select(.window=="SESSION:WIN")'`
> and `gh pr view PR_NUMBER --json title,body,headRefName` to reorient.
> Output each completed step as `CHECKPOINT:<step-name>` on its own line.

## Passing images and screenshots to agents

`tmux send-keys` is text-only — you cannot paste a raw image into a pane. To give an agent visual context (screenshots, diagrams, mockups):

1. **Save the image to a temp file** with a stable path:
   ```bash
   # If the user drags in a screenshot or you receive a file path:
   IMAGE_PATH="/tmp/orchestrator-context-$(date +%s).png"
   cp "$USER_PROVIDED_PATH" "$IMAGE_PATH"
   ```

2. **Reference the path in the objective string**:
   ```bash
   OBJECTIVE="Implement the layout shown in /tmp/orchestrator-context-1234567890.png. Read that image first with the Read tool to understand the design."
   ```

3. The agent uses its `Read` tool to view the image at startup — Claude Code agents are multimodal and can read image files directly.

**Rule**: always use `/tmp/orchestrator-context-<timestamp>.png` as the naming convention so the supervisor knows what to look for if it needs to re-brief an agent with the same image.

---

## Orchestrator final evaluation (YOU decide, not the script)

`verify-complete.sh` is a gate — it blocks premature marking. But it cannot tell you if the work is actually good. That is YOUR job.

When run-loop marks an agent `pending_evaluation` and you're notified, do all of these before marking done:

### 1. Run /pr-test (required, serialized, use TodoWrite to queue)

`/pr-test` is the only reliable confirmation that the objective is actually met. Run it yourself, not the agent.

**When multiple PRs reach `pending_evaluation` at the same time, use TodoWrite to queue them:**
```
- [ ] /pr-test PR #12636 — fix copilot retry logic
- [ ] /pr-test PR #12699 — builder chat panel
```
Run one at a time. Check off as you go.

```
/pr-test https://github.com/Significant-Gravitas/AutoGPT/pull/PR_NUMBER
```

**/pr-test can be lazy** — if it gives vague output, re-run with full context:

```
/pr-test https://github.com/OWNER/REPO/pull/PR_NUMBER
Context: This PR implements <objective from state file>. Key files: <list>.
Please verify: <specific behaviors to check>.
```

Only one `/pr-test` at a time — they share ports and DB.

### /pr-test result evaluation

**PARTIAL on any headline feature scenario is an immediate blocker.** Do not approve, do not mark done, do not let the agent output `ORCHESTRATOR:DONE`.

| `/pr-test` result | Action |
|---|---|
| All headline scenarios **PASS** | Proceed to evaluation step 2 |
| Any headline scenario **PARTIAL** | Re-brief the agent immediately — see below |
| Any headline scenario **FAIL** | Re-brief the agent immediately |

**What PARTIAL means**: the feature is only partly working. Example: the Apply button never appeared, or the AI returned no action blocks. The agent addressed part of the objective but not all of it.

**When any headline scenario is PARTIAL or FAIL:**

1. Do NOT mark the agent done or accept `ORCHESTRATOR:DONE`
2. Re-brief the agent with the specific scenario that failed and what was missing:
   ```bash
   tmux send-keys -t SESSION:WIN "PARTIAL result on /pr-test — S5 (Apply button) never appeared. The AI must output JSON action blocks for the Apply button to render. Fix this before re-running /pr-test."
   sleep 0.3
   tmux send-keys -t SESSION:WIN Enter
   ```
3. Set state back to `running`:
   ```bash
   jq --arg w "SESSION:WIN" '(.agents[] | select(.window == $w)).state = "running"' \
     ~/.claude/orchestrator-state.json > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
   ```
4. Wait for new `ORCHESTRATOR:DONE`, then re-run `/pr-test` from scratch

**Rule: only ALL-PASS qualifies for approval.** A mix of PASS + PARTIAL is a failure.

> **Why this matters**: PR #12699 was wrongly approved with S5 PARTIAL — the AI never output JSON action blocks so the Apply button never appeared. The fix was already in the agent's reach but slipped through because PARTIAL was not treated as blocking.

### 2. Do your own evaluation

1. **Read the PR diff and objective** — does the code actually implement what was asked? Is anything obviously missing or half-done?
2. **Read the resolved threads** — were comments addressed with real fixes, or just dismissed/resolved without changes?
3. **Check CI run names** — any suspicious retries that shouldn't have passed?
4. **Check the PR description** — title, summary, test plan complete?

### 3. Decide

- `/pr-test` all scenarios PASS + evaluation looks good → mark `done` in state, tell the user the PR is ready, ask if window should be closed
- `/pr-test` any scenario PARTIAL or FAIL → re-brief the agent with the specific failing scenario, set state back to `running` (see `/pr-test result evaluation` above)
- Evaluation finds gaps even with all PASS → re-brief the agent with specific gaps, set state back to `running`

**Never mark done based purely on script output.** You hold the full objective context; the script does not.

```bash
# Mark done after your positive evaluation:
jq --arg w "SESSION:WIN" '(.agents[] | select(.window == $w)).state = "done"' \
  ~/.claude/orchestrator-state.json > /tmp/orch.tmp && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json
```

## When to stop the fleet

Stop the fleet (`active = false`) when **all** of the following are true:

| Check | How to verify |
|---|---|
| All agents are `done` or `escalated` | `jq '[.agents[] | select(.state | test("running\|stuck\|idle\|waiting_approval"))] | length' ~/.claude/orchestrator-state.json` == 0 |
| All PRs have 0 unresolved review threads | GraphQL `isResolved` check per PR |
| All PRs have green CI **on a run triggered after the agent's last push** | `gh run list --branch BRANCH --limit 1` timestamp > `spawned_at` in state |
| No fresh CHANGES_REQUESTED (after latest commit) | `verify-complete.sh` checks this — stale pre-commit reviews are ignored |
| No agents are `escalated` without human review | If any are escalated, surface to user first |

**Do NOT stop just because agents output `ORCHESTRATOR:DONE`.** That is a signal to verify, not a signal to stop.

**Do stop** if the user explicitly says "stop", "shut down", or "kill everything", even with agents still running.

```bash
# Graceful stop
jq '.active = false' ~/.claude/orchestrator-state.json > /tmp/orch.tmp \
  && mv /tmp/orch.tmp ~/.claude/orchestrator-state.json

LOOP_WINDOW=$(jq -r '.loop_window // ""' ~/.claude/orchestrator-state.json)
[ -n "$LOOP_WINDOW" ] && tmux kill-window -t "$LOOP_WINDOW" 2>/dev/null || true
```

Does **not** recycle running worktrees — agents may still be mid-task. Run `capacity.sh` to see what's still in progress.

## tmux send-keys pattern

**Always split long messages into text + Enter as two separate calls with a sleep between them.** If sent as one call (`"text" Enter`), Enter can fire before the full string is buffered into Claude's input — leaving the message stuck as `[Pasted text +N lines]` unsent.

```bash
# CORRECT — text then Enter separately
tmux send-keys -t "$WINDOW" "your long message here"
sleep 0.3
tmux send-keys -t "$WINDOW" Enter

# WRONG — Enter may fire before text is buffered
tmux send-keys -t "$WINDOW" "your long message here" Enter
```

Short single-character sends (`y`, `Down`, empty Enter for dialog approval) are safe to combine since they have no buffering lag.

---

## Protected worktrees

Some worktrees must **never** be used as spare worktrees for agent tasks because they host files critical to the orchestrator itself:

| Worktree | Protected branch | Why |
|---|---|---|
| `AutoGPT1` | `dx/orchestrate-skill` | Hosts the orchestrate skill scripts. `recycle-agent.sh` would check out `spare/1`, wiping `.claude/skills/` and breaking all subsequent `spawn-agent.sh` calls. |

**Rule**: when selecting spare worktrees via `find-spare.sh`, skip any worktree whose CURRENT branch matches a protected branch. If you accidentally spawn an agent in a protected worktree, do not let `recycle-agent.sh` run on it — manually restore the branch after the agent finishes.

When `dx/orchestrate-skill` is merged into `dev`, `AutoGPT1` becomes a normal spare again.

---

## Thread resolution integrity (critical)

**Agents MUST NOT resolve review threads via GraphQL unless a real code fix has been committed and pushed first.**

This is the most common failure mode: agents call `resolveReviewThread` to make unresolved counts drop without actually fixing anything. This produces a false "done" signal that gets past verify-complete.sh.

**The only valid resolution sequence:**
1. Read the thread and understand what it's asking
2. Make the actual code change
3. `git commit` and `git push`
4. Reply to the thread with the commit SHA (e.g. "Fixed in `abc1234`")
5. THEN call `resolveReviewThread`

**The supervisor must verify actual thread counts via GraphQL** — never trust an agent's claim of "0 unresolved." After any agent's ORCHESTRATOR:DONE, always run:

```bash
# Step 1: get total count
TOTAL=$(gh api graphql -f query='{ repository(owner: "OWNER", name: "REPO") { pullRequest(number: PR) { reviewThreads { totalCount } } } }' \
  | jq '.data.repository.pullRequest.reviewThreads.totalCount')
echo "Total threads: $TOTAL"

# Step 2: paginate all pages and count unresolved
CURSOR=""; UNRESOLVED=0
while true; do
  AFTER=${CURSOR:+", after: \"$CURSOR\""}
  PAGE=$(gh api graphql -f query="{ repository(owner: \"OWNER\", name: \"REPO\") { pullRequest(number: PR) { reviewThreads(first: 100${AFTER}) { pageInfo { hasNextPage endCursor } nodes { isResolved } } } } }")
  UNRESOLVED=$(( UNRESOLVED + $(echo "$PAGE" | jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved==false)] | length') ))
  HAS_NEXT=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
  CURSOR=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.endCursor')
  [ "$HAS_NEXT" = "false" ] && break
done
echo "Unresolved: $UNRESOLVED"
```

If unresolved > 0, the agent is NOT done — re-brief with the actual count and the rule.

**Include this in every agent objective:**
> IMPORTANT: Do NOT resolve any review thread via GraphQL unless the code fix is committed and pushed first. Fix the code → commit → push → reply with SHA → then resolve. Never resolve without a real commit. "Accepted" or "Acknowledged" replies are NOT resolutions — only real commits qualify.

### Detecting fake resolutions

When an agent claims "0 unresolved threads", query GitHub GraphQL yourself and also inspect how each thread was resolved. A resolved thread whose last comment is `"Acknowledged"`, `"Same as above"`, `"Accepted trade-off"`, or `"Deferred"` — with no commit SHA — is a fake resolution.

To spot these, paginate all pages and collect resolved threads with missing SHA links:
```bash
# Paginate all pages — first:100 misses threads beyond page 1 on large PRs
CURSOR=""; FAKE_RESOLUTIONS="[]"
while true; do
  AFTER=${CURSOR:+", after: \"$CURSOR\""}
  PAGE=$(gh api graphql -f query="
  {
    repository(owner: \"Significant-Gravitas\", name: \"AutoGPT\") {
      pullRequest(number: PR_NUMBER) {
        reviewThreads(first: 100${AFTER}) {
          pageInfo { hasNextPage endCursor }
          nodes {
            isResolved
            comments(last: 1) {
              nodes { body author { login } }
            }
          }
        }
      }
    }
  }")
  PAGE_FAKES=$(echo "$PAGE" | jq '[.data.repository.pullRequest.reviewThreads.nodes[]
    | select(.isResolved == true)
    | {body: .comments.nodes[0].body[:120], author: .comments.nodes[0].author.login}
    | select(.body | test("Fixed in|Removed in|Addressed in") | not)]')
  FAKE_RESOLUTIONS=$(echo "$FAKE_RESOLUTIONS $PAGE_FAKES" | jq -s 'add')
  HAS_NEXT=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
  CURSOR=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.endCursor')
  [ "$HAS_NEXT" = "false" ] && break
done
echo "$FAKE_RESOLUTIONS"
```
Any resolved thread whose last comment does NOT contain `"Fixed in"`, `"Removed in"`, or `"Addressed in"` (with a commit link) should be investigated — either the agent falsely resolved it, or it was a genuine false positive that needs explanation.

## GitHub abuse rate limits

Two distinct rate limits exist with different recovery times:

| Error | HTTP status | Cause | Recovery |
|---|---|---|---|
| `{"code":"abuse"}` in body | 403 | Secondary rate limit — too many write operations (comments, mutations) in a short window | Wait **2–3 minutes**. 60s is often not enough. |
| `API rate limit exceeded` | 429 | Primary rate limit — too many read calls per hour | Wait until `X-RateLimit-Reset` timestamp |

**Prevention:** Agents must add `sleep 3` between individual thread reply API calls. For >20 unresolved threads, increase to `sleep 5`.

If you see a 403 `abuse` error from an agent's pane:
1. Nudge the agent: `"You hit a GitHub secondary rate limit (403). Stop all API writes. Wait 2 minutes, then resume with sleep 3 between each thread reply."`
2. Do NOT nudge again during the 2-minute wait — a second nudge restarts the clock.

Add this to agent briefings when there are >20 unresolved threads:
> Post replies with `sleep 3` between each reply. If you hit a 403 abuse error, wait 2 minutes (not 60s — secondary limits take longer to clear) then continue.

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
12. **ORCHESTRATOR:DONE is a signal to verify, not to accept** — always run `verify-complete.sh` and check CI run timestamp before recycling
13. **Protected worktrees** — never use the worktree hosting the skill scripts as a spare
14. **Images via file path** — save screenshots to `/tmp/orchestrator-context-<ts>.png`, pass path in objective; agents read with the `Read` tool
15. **Split send-keys** — always separate text and Enter with `sleep 0.3` between calls for long strings
16. **Poll ALL windows from state file** — never hardcode window count. Derive active windows dynamically: `jq -r '.agents[] | select(.state | test("running|idle|stuck")) | .window' ~/.claude/orchestrator-state.json`. If you added a window mid-session outside spawn-agent.sh, add it to the state file immediately.
20. **Orchestrator handles its own approvals** — when spawning a subagent to make edits (SKILL.md, scripts, config), review the diff yourself and approve/reject without surfacing it to the user. The user should never have to open a file to check the orchestrator's work. Use the Agent tool with `subagent_type: general-purpose` for drafting, then verify the result yourself before considering the task done.
17. **Update state file on re-task** — whenever an agent is re-tasked mid-session (objective changes, new PR assigned), update the state file record immediately so objectives stay accurate for re-briefing after compaction.
18. **No GraphQL resolveReviewThread without a commit** — see Thread resolution integrity above. This is rule #1 for pr-address work.
19. **Verify thread counts yourself** — after any agent claims "0 unresolved threads", query GitHub GraphQL directly before accepting. Never trust the agent's self-report.
