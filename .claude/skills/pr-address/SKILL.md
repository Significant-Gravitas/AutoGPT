---
name: pr-address
description: Address PR review comments and loop until CI green and all comments resolved. TRIGGER when user asks to address comments, fix PR feedback, respond to reviewers, or babysit/monitor a PR.
user-invocable: true
argument-hint: "[PR number or URL] — if omitted, finds PR for current branch."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# PR Address

## Find the PR

```bash
gh pr list --head $(git branch --show-current) --repo Significant-Gravitas/AutoGPT
gh pr view {N}
```

## Fetch comments (all sources)

### 1. Inline review threads — GraphQL (primary source of actionable items)

Use GraphQL to fetch inline threads. It natively exposes `isResolved`, returns threads already grouped with all replies, and paginates via cursor — no manual thread reconstruction needed.

```bash
gh api graphql -f query='
{
  repository(owner: "Significant-Gravitas", name: "AutoGPT") {
    pullRequest(number: {N}) {
      reviewThreads(first: 100) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          path
          comments(last: 1) {
            nodes { databaseId body author { login } createdAt }
          }
        }
      }
    }
  }
}'
```

If `pageInfo.hasNextPage` is true, fetch subsequent pages by adding `after: "<endCursor>"` to `reviewThreads(first: 100, after: "...")` and repeat until `hasNextPage` is false.

**Filter to unresolved threads only** — skip any thread where `isResolved: true`. `comments(last: 1)` returns the most recent comment in the thread — act on that; it reflects the reviewer's final ask. Use the thread `id` (Relay global ID) to track threads across polls.

### 2. Top-level reviews — REST (MUST paginate)

```bash
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews --paginate
```

**CRITICAL — always `--paginate`.** Reviews default to 30 per page. PRs can have 80–170+ reviews (mostly empty resolution events). Without pagination you miss reviews past position 30 — including `autogpt-reviewer`'s structured review which is typically posted after several CI runs and sits well beyond the first page.

Two things to extract:
- **Overall state**: look for `CHANGES_REQUESTED` or `APPROVED` reviews.
- **Actionable feedback**: non-empty bodies only. Empty-body reviews are thread-resolution events — they indicate progress but have no feedback to act on.

**Where each reviewer posts:**
- `autogpt-reviewer` — posts detailed structured reviews ("Blockers", "Should Fix", "Nice to Have") as **top-level reviews**. Not present on every PR. Address ALL items.
- `sentry[bot]` — posts bug predictions as **inline threads**. Fix real bugs, explain false positives.
- `coderabbitai[bot]` — posts summaries as **top-level reviews** AND actionable items as **inline threads**. Address actionable items.
- Human reviewers — can post in any source. Address ALL non-empty feedback.

### 3. PR conversation comments — REST

```bash
gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments --paginate
```

Mostly contains: bot summaries (`coderabbitai[bot]`), CI/conflict detection (`github-actions[bot]`), and author status updates. Scan for non-empty messages from non-bot human reviewers that aren't the PR author — those are the ones that need a response.

## For each unaddressed comment

Address comments **one at a time**: fix → commit → push → inline reply → next.

1. Read the referenced code, make the fix (or reply explaining why it's not needed)
2. Commit and push the fix
3. Reply **inline** (not as a new top-level comment) referencing the fixing commit — this is what resolves the conversation for bot reviewers (coderabbitai, sentry):

| Comment type | How to reply |
|---|---|
| Inline review (`pulls/{N}/comments`) | `gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments/{ID}/replies -f body="🤖 Fixed in <commit-sha>: <description>"` |
| Conversation (`issues/{N}/comments`) | `gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments -f body="🤖 Fixed in <commit-sha>: <description>"` |

## Format and commit

After fixing, format the changed code:

- **Backend** (from `autogpt_platform/backend/`): `poetry run format`
- **Frontend** (from `autogpt_platform/frontend/`): `pnpm format && pnpm lint && pnpm types`

If API routes changed, regenerate the frontend client:
```bash
cd autogpt_platform/backend && poetry run rest &
REST_PID=$!
trap "kill $REST_PID 2>/dev/null" EXIT
WAIT=0; until curl -sf http://localhost:8006/health > /dev/null 2>&1; do sleep 1; WAIT=$((WAIT+1)); [ $WAIT -ge 60 ] && echo "Timed out" && exit 1; done
cd ../frontend && pnpm generate:api:force
kill $REST_PID 2>/dev/null; trap - EXIT
```
Never manually edit files in `src/app/api/__generated__/`.

Then commit and **push immediately** — never batch commits without pushing.

For backend commits in worktrees: `poetry run git commit` (pre-commit hooks).

## The loop

```text
address comments → format → commit → push
→ wait for CI (while addressing new comments) → fix failures → push
→ re-check comments after CI settles
→ repeat until: all comments addressed AND CI green AND no new comments arriving
```

### Polling for CI + new comments

After pushing, poll for **both** CI status and new comments in a single loop. Do not use `gh pr checks --watch` — it blocks the tool and prevents reacting to new comments while CI is running.

> **Note:** `gh pr checks --watch --fail-fast` is tempting but it blocks the entire Bash tool call, meaning the agent cannot check for or address new comments until CI fully completes. Always poll manually instead.

**Polling loop — repeat every 30 seconds:**

1. Check CI status:
```bash
gh pr checks {N} --repo Significant-Gravitas/AutoGPT --json bucket,name,link
```
   Parse the results: if every check has `bucket` of `"pass"` or `"skipping"`, CI is green. If any has `"fail"`, CI has failed. Otherwise CI is still pending.

2. Check for merge conflicts:
```bash
gh pr view {N} --repo Significant-Gravitas/AutoGPT --json mergeable --jq '.mergeable'
```
   If the result is `"CONFLICTING"`, the PR has a merge conflict — see "Resolving merge conflicts" below. If `"UNKNOWN"`, GitHub is still computing mergeability — wait and re-check next poll.

3. Check for new/changed comments (all three sources):

   **Inline threads** — re-run the GraphQL query from "Fetch comments". For each unresolved thread, record `{thread_id, last_comment_databaseId}` as your baseline. On each poll, action is needed if:
   - A new thread `id` appears that wasn't in the baseline (new thread), OR
   - An existing thread's `last_comment_databaseId` has changed (new reply on existing thread)

   **Conversation comments:**
   ```bash
   gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments --paginate
   ```
   Compare total count and newest `id` against baseline. Filter to non-empty, non-bot, non-author-update messages.

   **Top-level reviews:**
   ```bash
   gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews --paginate
   ```
   Watch for new non-empty reviews (`CHANGES_REQUESTED` or `COMMENTED` with body). Compare total count and newest `id` against baseline.

4. **React in this precedence order (first match wins):**

| What happened | Action |
|---|---|
| Merge conflict detected | See "Resolving merge conflicts" below. |
| Mergeability is `UNKNOWN` | GitHub is still computing mergeability. Sleep 30 seconds, then restart polling from the top. |
| New comments detected | Address them (fix → commit → push → reply). After pushing, re-fetch all comments to update your baseline, then restart this polling loop from the top (new commits invalidate CI status). |
| CI failed (bucket == "fail") | Get failed check links: `gh pr checks {N} --repo Significant-Gravitas/AutoGPT --json bucket,link --jq '.[] \| select(.bucket == "fail") \| .link'`. Extract run ID from link (format: `.../actions/runs/<run-id>/job/...`), read logs with `gh run view <run-id> --repo Significant-Gravitas/AutoGPT --log-failed`. Fix → commit → push → restart polling. |
| CI green + no new comments | **Do not exit immediately.** Bots (coderabbitai, sentry) often post reviews shortly after CI settles. Continue polling for **2 more cycles (60s)** after CI goes green. Only exit after 2 consecutive green+quiet polls. |
| CI pending + no new comments | Sleep 30 seconds, then poll again. |

**The loop ends when:** CI fully green + all comments addressed + **2 consecutive polls with no new comments after CI settled.**

### Resolving merge conflicts

1. Identify the PR's target branch and remote:
```bash
gh pr view {N} --repo Significant-Gravitas/AutoGPT --json baseRefName --jq '.baseRefName'
git remote -v   # find the remote pointing to Significant-Gravitas/AutoGPT (typically 'upstream' in forks, 'origin' for direct contributors)
```

2. Pull the latest base branch with a 3-way merge:
```bash
git pull {base-remote} {base-branch} --no-rebase
```

3. Resolve conflicting files, then verify no conflict markers remain:
```bash
if grep -R -n -E '^(<<<<<<<|=======|>>>>>>>)' <conflicted-files>; then
  echo "Unresolved conflict markers found — resolve before proceeding."
  exit 1
fi
```

4. Stage and push:
```bash
git add <conflicted-files>
git commit -m "Resolve merge conflicts with {base-branch}"
git push
```

5. Restart the polling loop from the top — new commits reset CI status.
