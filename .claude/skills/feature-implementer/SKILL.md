---
name: feature-implementer
description: "End-to-end AutoGPT Platform implementation pipeline: plan, review-plan, implement, review-impl, commit — each step run in a fresh spawned agent. TRIGGER when user asks to implement a feature/fix end-to-end, run the implementation pipeline, or build something 'with the full loop'."
user-invocable: true
args: "[task description] — feature, bug fix, or enhancement; or a pre-existing plan."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Feature Implementer (Orchestrator)

This is the orchestrator for the AutoGPT Platform implementation pipeline. It runs as a **skill in the main thread** so it can spawn agents for every step that benefits from fresh context (planning, plan review, surgical implementation, implementation review). Do not turn this into an agent — agents cannot spawn sub-agents, which silently degrades the pipeline into self-review.

For new blocks or provider integrations, use `/block-implementer` instead — it is this pipeline specialized for blocks.

## Roles

| Step | Where it runs | Why |
|---|---|---|
| 1. Produce plan | **Spawned `general-purpose` agent** invoking `/feature-planner` | Fresh context = plan shaped by the task, not by the conversation that led here |
| 2. Review plan | **Spawned `general-purpose` agent** invoking `/review-feature-plan` | Honest architectural review, independent of the planner |
| 3. Implement | **Spawned `implementation-executor` agent** | Surgical edits + test-first verification |
| 4. Spot-check verification | This thread | Re-run anything the executor skipped |
| 5. Review implementation | **Spawned `general-purpose` agent** invoking `/review-impl` | Independent reviewer, not the implementer |
| 6. Commit | This thread | Owner of the working tree decides what gets staged |
| 7. Ship (when requested) | This thread, invoking `/open-pr` then `/pr-polish` | `/pr-polish` must run in the foreground — spawned agents can't invoke child skills |

The orchestrator never authors content itself. Its only jobs are: spawn agents, route their output to the next step, loop review steps until clean, and own the commit.

When spawning the executor, prefer the strongest available model: request a `fable` model override; if the harness rejects it, spawn without an override (the agent's frontmatter falls back to `opus`).

## Inputs

Either:

1. A task description (feature, bug, affected subsystems, expected behavior), or
2. A pre-existing plan — treat as a draft unless it has already passed `/review-feature-plan` to clean.

If isolation is needed (other agents active in this worktree), prepare a worktree first (`/branchlet`) and pass its path to the executor.

## Pipeline

### Step 1 — Produce the plan

Spawn a `general-purpose` agent and instruct it to invoke `/feature-planner`.

**Spawn inputs:** task description; in-scope file/subsystem hints; any prior reviewer findings (none on first round).

Do not author or edit the plan in this thread. If the returned plan is missing sections or is superficial, send the same inputs plus an explicit "missing sections" note to a **fresh** planning agent — do not patch it yourself.

### Step 2 — Review the plan until clean (unbounded loop)

Spawn a `general-purpose` agent and instruct it to invoke `/review-feature-plan` against the full plan.

**Reviewer spawn inputs:** the full plan; the original task description.

If the reviewer returns findings, spawn a **fresh** planning agent (Step 1 inputs plus the reviewer's findings as additional constraints), then a **fresh** reviewer against the revised plan.

**Repeat until a full review round returns zero findings.** There is no iteration cap — "two rounds and ship" is not acceptable. Stop only for:

- a true product/design decision the planner cannot resolve (ask the user),
- missing external access (API docs unavailable, credentials missing), or
- an environment blocker that makes review impossible.

Each review runs in a fresh agent context — never reuse the previous reviewer's context.

### Step 3 — Dispatch implementation

Spawn the `implementation-executor` agent.

**Spawn inputs:** the reviewed clean plan in full; in-bounds / out-of-bounds file scope; worktree path if applicable.

The executor edits test-first, runs targeted verification, and returns a structured report (diff summary, test-first evidence, verification results, judgement calls, stop-and-return items, deviations, risks).

If the executor returns stop-and-return items (plan contradicts current code, auth implications unclear, suppressor would be needed), do NOT improvise around them. Loop back to Step 1, feed the executor's findings into `/feature-planner` as new constraints, and re-run Steps 1–3.

### Step 4 — Spot-check verification

Re-run only what the executor skipped or what changed because of intervening commits. **Never run two pytest invocations concurrently — they exhaust database connection slots.** Always confirm formatting before review:

```bash
poetry run format    # backend changes (run from autogpt_platform/backend)
pnpm format          # frontend changes (run from autogpt_platform/frontend)
```

Decide here whether the diff warrants a full `poetry run test` / `pnpm test:unit` pass beyond the executor's targeted runs (cross-cutting changes do; localized changes usually don't — CI covers the rest).

### Step 5 — Review implementation until clean (unbounded loop)

Spawn a `general-purpose` agent and instruct it to invoke `/review-impl` against the implementation diff. The reviewer MUST verify the originally reported bug or requirement via a discriminating test — not just that the code looks clean.

**Reviewer spawn inputs:** `git diff` of the branch against its base; the original task description; the reviewed plan.

If the reviewer returns findings, spawn a **fresh** `implementation-executor` agent to apply fixes:

**Fix-round spawn inputs:** the reviewed plan; current `git diff HEAD`; the reviewer findings as fix constraints; same scope as Step 3.

Then spawn a **fresh** review agent against the new diff. **Repeat until a full review round returns zero findings.** No iteration cap. Never self-review — always spawn an isolated reviewer.

### Step 6 — Commit

Commit only after the Step 2 plan loop is clean, Step 4 verification passes, and the Step 5 review loop is clean.

Stage by pathspec — never `git add -A` and never commit without a pathspec; other agents may have staged work in the shared tree:

```bash
git status --short
git diff --stat
git commit <paths> -m "<type>(<scope>): <summary>"   # conventional commit, repo scopes
```

Verify HEAD is on a branch (not detached) before any push. Do not push or open a PR unless asked — when asked, proceed to Step 7.

### Step 7 — Ship (when requested)

The pipeline's review loops gate the **working tree**; the PR surface has its own review stack. Chain into it rather than duplicating it:

1. For security-sensitive diffs (auth, credentials, schema broadening, file handling), run `/security-review` against the branch first — cheaper to fix before reviewers and bots see it.
2. Invoke `/open-pr` — it owns pre-flight checks, the PR template, and the `dev` base.
3. Invoke `/pr-polish` on the new PR — it alternates `/pr-review` and `/pr-address` until merge-ready (zero new findings, all threads resolved, CI green, two clean polls).

Both `/open-pr` and `/pr-polish` run **inline in this thread** — never spawn `/pr-polish` into a background or sub-agent; spawned agents don't inherit the skill registry, so its child `Skill()` calls silently fail (documented in `/pr-polish` itself).

Division of labor: `/review-impl` (Step 5) is the pre-commit gate on the diff; `/pr-review` is the PR-surface gate layered with bot reviews; `/pr-polish` loops the latter with `/pr-address`. If a `/pr-review` round surfaces a plan-level problem (wrong approach, missing class coverage), don't patch it in the address loop — return to Step 1 with the finding as a new constraint.

## Final Report

Return after the commit:

1. Plan-review rounds (count) and final clean result.
2. What changed, grouped by subsystem and file.
3. Key architectural decisions.
4. Verification commands run and results (executor's + spot-checks).
5. Implementation-review rounds (count) and final clean result.
6. Commit hash and staged file list.
7. If shipped: PR URL and `/pr-polish` outcome (rounds, final state).
8. Deviations from the plan with reasons.
9. Self-flagged risks and judgement calls (yours + executor's).
10. Remaining items, if any, with reasons.
