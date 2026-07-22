---
name: ship-issue
description: End-to-end issue shipping loop — takes a Linear issue link, fixes the issue, opens a PR, drives CI to green, addresses all review comments, deploys a preview via !deploy, verifies the fix on the preview, and reports what was fixed and how to replicate it. TRIGGER when the user gives a Linear issue link/ID and wants it fixed and shipped, or says "ship this issue", "fix and deploy this ticket", "take this issue to preview", or wants the full fix→PR→CI→review→deploy pipeline run autonomously.
user-invocable: true
argument-hint: "<Linear issue URL or ID, e.g. https://linear.app/autogpt/issue/OPEN-1234/... or OPEN-1234>"
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Ship Issue

One invocation drives a Linear issue all the way to a verified preview deployment:

```text
fetch issue → fix on a branch → /open-pr → /pr-address (CI green + comments resolved)
→ comment !deploy → wait for preview → verify fix on preview → final report
```

This is an **orchestration skill**: delegate the heavy lifting to `/open-pr` and `/pr-address` via the `Skill` tool — do not re-implement their logic. Like `/pr-polish`, run it **in the foreground** (child `Skill()` calls are not available to background agents), and **never end your response between stages** — a child skill returning is a loop-iteration boundary, not a conversation-turn boundary. Keep going until the final report or a genuine blocker that needs the user.

## Stage 0 — Intake the Linear issue

Parse the argument: a Linear URL (`https://linear.app/{org}/issue/{ID}/{slug}` → extract `{ID}`, e.g. `OPEN-1234`) or a bare ID.

Fetch the issue with the Linear MCP tool `mcp__Linear__get_issue` (load via ToolSearch if not yet loaded). Also fetch its comments (`mcp__Linear__list_comments`) — repro details and constraints often live there, not in the description.

From the issue, before writing any code, write down:

1. **Expected vs actual behavior** — what the bug/feature actually is.
2. **Repro steps in dev** — the concrete click-path or API calls that demonstrate the problem. If the issue doesn't state them, derive them from the code while investigating. These steps are load-bearing twice later: they go in the final report ("how to replicate in dev") and they are the script for preview verification in Stage 5. Getting them precise now prevents a vague verification later.
3. **Definition of done** — what observable change proves the fix.

If the issue is too vague to derive expected behavior at all, stop and ask the user — that's the one thing worth blocking on.

Move the Linear issue to **In Progress** (`mcp__Linear__save_issue` with the appropriate state) so the team sees it's being worked.

## Stage 1 — Branch and fix

1. Start from the latest default branch:
   ```bash
   git fetch origin dev && git checkout -B {branch-name} origin/dev
   ```
   Use Linear's suggested git branch name for the issue if available (the issue payload includes it), otherwise `{user}/{ticket-id-lowercase}-{short-slug}` per repo convention (e.g. `abhi/open-1234-fix-run-resume`).
2. Investigate to the **root cause** — read the failing path end-to-end before editing. A fix that patches the symptom will bounce in review.
3. Implement the minimal fix. Add/update tests covering the new behavior (Codecov patch target is 80% on changed lines — this bites in Stage 3 if skipped now).
4. Format and lint per codebase:
   - Backend: `poetry run format` (from `autogpt_platform/backend/`)
   - Frontend: `pnpm format && pnpm lint && pnpm types` (from `autogpt_platform/frontend/`)
5. Run the tests relevant to the touched modules locally.
6. Commit with a conventional message referencing the ticket, e.g. `fix(backend): resume dropped runs on executor restart (OPEN-1234)`.

## Stage 2 — Open the PR

Invoke the existing skill — do not hand-roll the PR:

```python
Skill(skill="open-pr")
```

Additions on top of what `/open-pr` does:

- Include the Linear magic words in the PR body's Why/What section — `Fixes OPEN-1234` — so Linear auto-links and auto-closes the ticket on merge.
- Post the PR link back to the Linear issue as a comment (`mcp__Linear__save_comment`) if the Linear↔GitHub integration hasn't already attached it.

## Stage 3 — CI green + all comments addressed

Invoke:

```python
Skill(skill="pr-address", args=pr_url)
```

`/pr-address` owns the whole loop: polling CI, fixing failures, fetching all comment sources (paginated), fix → commit → push → reply → resolve, and exiting only after CI is green with 2 consecutive quiet polls. Trust its exit condition, but **re-verify independently** before moving on (never trust a summary for a gate):

```bash
gh pr checks {N} --repo Significant-Gravitas/AutoGPT --json bucket \
  | jq '[.[] | select(.bucket != "pass" and .bucket != "skipping")] | length'   # must be 0
```

plus the paginated unresolved-thread count from `pr-address`'s verification section (must be 0). If either check fails, re-invoke `/pr-address` — do not proceed to deploy on a red or contested PR.

## Stage 4 — Deploy the preview

Gate: CI fully green, 0 unresolved threads, no unaddressed reviews. Only then:

```bash
gh pr comment {N} --repo Significant-Gravitas/AutoGPT --body '!deploy'
```

The comment body must be **exactly** `!deploy` and nothing else — the dispatcher workflow (`platform-dev-deploy-event-dispatcher.yml`) does an exact match on the trimmed body; any extra text silently does nothing. The commenter must be an org member/collaborator; if a `❌ Permission denied` comment appears, stop and tell the user to post `!deploy` themselves.

Then poll for deployment progress every 60s (comments + deployment statuses):

```bash
# 1. Dispatcher ack — a "🚀 Deploying PR #N..." comment should appear within ~2 min
gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments --paginate \
  --jq '[.[] | select(.body | test("Deploying PR|deployed|preview"; "i")) | {user: .user.login, body: .body[:300], created_at}]'

# 2. GitHub deployment statuses (the infra repo reports here when it finishes)
gh api repos/Significant-Gravitas/AutoGPT/deployments --jq '[.[] | select(.ref == "{branch}")][0].id' \
  | xargs -I{} gh api repos/Significant-Gravitas/AutoGPT/deployments/{}/statuses \
      --jq '[.[0] | {state, environment_url, description}]'
```

Capture the **preview URL** from whichever source posts it (bot comment or `environment_url`). Notes:

- The actual deploy runs in `AutoGPT_cloud_infrastructure` (separate repo) — you cannot watch its logs, only the signals it sends back to the PR. First deploys typically take 10–20 minutes.
- If no ack comment appears within 5 minutes, the dispatcher didn't fire — check the comment body was exact and re-post once.
- If 30 minutes pass with no success signal, report the timeout to the user with what you observed instead of claiming success. Do not fabricate a preview URL.
- New pushes after `!deploy` auto-redeploy the preview (the dispatcher handles `synchronize`), so late fixes don't need a second `!deploy` — but they do reset the wait.

## Stage 5 — Verify the fix on the preview

This is the point of the whole loop — a deployed-but-unverified preview is not "done".

1. Take the **repro steps from Stage 0** and execute them against the preview URL.
2. Prefer a real browser (agent-browser / Playwright with the pre-installed Chromium) for UI issues — screenshot the before-broken-now-fixed state. For API/backend issues, `curl` the relevant endpoints and capture responses.
3. Log in with the dev test credentials used by `/pr-test` if auth is required.
4. Compare against the **definition of done** from Stage 0. The verdict is one of:
   - **Verified fixed** — repro steps no longer produce the bug; evidence captured.
   - **Not fixed / regressed** — go back to Stage 1 with the new evidence; the loop is not done.
   - **Could not verify** (auth wall, preview flaky, needs data you can't create) — say exactly what blocked verification and what the user should check manually. Never report "deployed" as if it meant "verified".

## Stage 6 — Final report

Move the Linear issue to **In Review** and comment the outcome on it. Then report to the user in this exact structure:

```markdown
## 🚢 OPEN-1234 shipped to preview

**PR**: {url} — CI green, {X} review comments addressed, all threads resolved
**Preview**: {preview-url} — deployed {time}, fix **verified** ✅ / **not verified** ⚠️ (reason)

### What was broken
{1–3 sentences: the root cause, in plain language}

### How it was fixed
{what changed, which files/modules, why this approach}

### How to replicate in dev
1. {exact steps to reproduce the original bug on a dev build — commands, URLs, clicks}
2. ...
3. Before this fix: {broken behavior}. After: {fixed behavior}.

### Preview verification
{what was checked on the preview, with evidence — screenshots / API responses}
```

## Failure handling

- **A stage fails repeatedly** (e.g. `/pr-address` hits its round cap, CI has an unrelated flake, deploy times out): stop looping, report exactly where the pipeline is stuck, what you tried, and the current PR state. A precise stuck-report is a valid outcome; silent stalling and false "done" are not.
- **Scope creep discovered mid-fix** (the "bug" is actually a large refactor): pause after Stage 0/1 and confirm scope with the user before opening a PR.
- **Never** comment `!deploy` while CI is red or threads are unresolved, and never mark the Linear issue Done — merge is a human decision; this skill ends at verified-preview.

## Related skills

| Skill | Role in this pipeline |
|---|---|
| `/open-pr` | Stage 2 — PR creation with template + review bot trigger |
| `/pr-address` | Stage 3 — CI + review-comment convergence loop |
| `/pr-test` | Optional pre-deploy local E2E if a docker workspace is available |
| `/pr-polish` | Substitute for Stage 3 when the user wants self-review rounds layered on top |
