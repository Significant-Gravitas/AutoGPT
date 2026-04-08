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

## Read the PR description

Understand the **Why / What / How** before addressing comments — you need context to make good fixes:

```bash
gh pr view {N} --json body --jq '.body'
```

## Fetch comments (all sources)

### 1. Inline review threads — GraphQL (primary source of actionable items)

> ⚠️ **WARNING — PAGINATE ALL PAGES BEFORE ADDRESSING ANYTHING**
>
> `reviewThreads(first: 100)` returns at most 100 threads per page. A PR with many review cycles can have 140+ threads across 2+ pages. **If you start addressing threads after fetching only page 1, you will miss all threads on subsequent pages and silently leave them unresolved.**
>
> PR #12636 had 142 total threads: page 1 returned 69 unresolved, page 2 had 42 more (111 total unresolved). An agent that stopped after page 1 addressed only 69 and falsely reported "done".
>
> **The rule: collect ALL thread IDs from ALL pages into a single list, then address them.**

**Step 1 — Fetch total count first:**

```bash
gh api graphql -f query='
{
  repository(owner: "Significant-Gravitas", name: "AutoGPT") {
    pullRequest(number: {N}) {
      reviewThreads { totalCount }
    }
  }
}' | jq '.data.repository.pullRequest.reviewThreads.totalCount'
```

If `totalCount > 100`, you have multiple pages. Fetch them all before doing anything else.

**Step 2 — Collect all unresolved thread IDs across all pages:**

```bash
# Accumulate all unresolved threads — loop until hasNextPage == false
CURSOR=""
ALL_THREADS="[]"
while true; do
  AFTER=${CURSOR:+", after: \"$CURSOR\""}
  PAGE=$(gh api graphql -f query="
  {
    repository(owner: \"Significant-Gravitas\", name: \"AutoGPT\") {
      pullRequest(number: {N}) {
        reviewThreads(first: 100${AFTER}) {
          pageInfo { hasNextPage endCursor }
          nodes {
            id
            isResolved
            path
            line
            comments(last: 1) {
              nodes { databaseId body author { login } }
            }
          }
        }
      }
    }
  }")
  # Append unresolved nodes from this page
  PAGE_THREADS=$(echo "$PAGE" | jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)]')
  ALL_THREADS=$(echo "$ALL_THREADS $PAGE_THREADS" | jq -s 'add')
  HAS_NEXT=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
  CURSOR=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.endCursor')
  [ "$HAS_NEXT" = "false" ] && break
done

echo "Total unresolved threads: $(echo "$ALL_THREADS" | jq 'length')"
echo "$ALL_THREADS" | jq '[.[] | {id, path, line, body: .comments.nodes[0].body[:200]}]'
```

**Step 3 — Address every thread in `ALL_THREADS`, then resolve.**

Only after this loop completes (all pages fetched, count confirmed) should you begin making fixes.

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

**CRITICAL: The only valid sequence is fix → commit → push → reply → resolve. Never resolve a thread without a real code commit.**

Resolving a thread via `resolveReviewThread` without an actual fix is the most common failure mode — it makes unresolved counts drop without any real change, producing a false "done" signal. If the issue was genuinely a false positive (no code change needed), reply explaining why and then resolve. Otherwise:

Address comments **one at a time**: fix → commit → push → inline reply → resolve.

1. Read the referenced code, make the fix (or reply explaining why it's not needed)
2. Commit and push the fix
3. Reply **inline** (not as a new top-level comment) referencing the fixing commit — this is what resolves the conversation for bot reviewers (coderabbitai, sentry):

Use a **markdown commit link** so GitHub renders it as a clickable reference. Always get the full SHA with `git rev-parse HEAD` **after** committing — never copy a SHA from a previous commit or hardcode one:

```bash
FULL_SHA=$(git rev-parse HEAD)
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments/{ID}/replies \
  -f body="🤖 Fixed in [${FULL_SHA:0:9}](https://github.com/Significant-Gravitas/AutoGPT/commit/${FULL_SHA}): <description>"
```

| Comment type | How to reply |
|---|---|
| Inline review (`pulls/{N}/comments`) | `gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments/{ID}/replies -f body="🤖 Fixed in [abc1234](https://github.com/Significant-Gravitas/AutoGPT/commit/FULL_SHA): <description>"` |
| Conversation (`issues/{N}/comments`) | `gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments -f body="🤖 Fixed in [abc1234](https://github.com/Significant-Gravitas/AutoGPT/commit/FULL_SHA): <description>"` |

### What counts as a valid resolution

Only two situations justify calling `resolveReviewThread`:

1. **Real code fix**: you changed the code, committed + pushed, and replied with the SHA. The commit diff must actually address the concern — not just touch the same file.
2. **Genuine false positive**: the reviewer's concern does not apply to this code, and you can give a specific technical reason (e.g. "Not applicable — `sdk_cwd` is pre-validated by `_make_sdk_cwd()` which applies normpath + prefix assertion before reaching this point").

**Anti-patterns that look resolved but aren't — never do these:**
- `"Accepted, tracked as follow-up"` — a deferral, not a fix. The concern is still open. Do not resolve.
- `"Acknowledged"` or `"Same as above"` — these are acknowledgements, not fixes. Do not resolve.
- `"Fixed in abc1234"` where `abc1234` is a commit that doesn't actually change the flagged line/logic — dishonest. Verify `git show abc1234 -- path/to/file` changes the right thing before posting.
- Resolving without replying — the reviewer never sees what happened.

When in doubt: if a code change is needed, make it. A deferred issue means the thread stays open until the follow-up PR is merged.

## Codecov coverage

Codecov patch target is **80%** on changed lines. Checks are **informational** (not blocking) but should be green.

### Running coverage locally

**Backend** (from `autogpt_platform/backend/`):
```bash
poetry run pytest -s -vv --cov=backend --cov-branch --cov-report term-missing
```

**Frontend** (from `autogpt_platform/frontend/`):
```bash
pnpm vitest run --coverage
```

### When codecov/patch fails

1. Find uncovered files: `git diff --name-only $(gh pr view --json baseRefName --jq '.baseRefName')...HEAD`
2. For each uncovered file — extract inline logic to `helpers.ts`/`helpers.py` and test those (highest ROI). Colocate tests as `*_test.py` (backend) or `__tests__/*.test.ts` (frontend).
3. Run coverage locally to verify, commit, push.

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

Then commit and **push immediately** — never batch commits without pushing. Each fix should be visible on GitHub right away so CI can start and reviewers can see progress.

**Never push empty commits** (`git commit --allow-empty`) to re-trigger CI or bot checks. When a check fails, investigate the root cause (unchecked PR checklist, unaddressed review comments, code issues) and fix those directly. Empty commits add noise to git history.

For backend commits in worktrees: `poetry run git commit` (pre-commit hooks).

## Coverage

Codecov enforces patch coverage on new/changed lines — new code you write must be tested. Before pushing, verify you haven't left new lines uncovered:

```bash
cd autogpt_platform/backend
poetry run pytest --cov=. --cov-report=term-missing {path/to/changed/module}
```

Look for lines marked `miss` — those are uncovered. Add tests for any new code you wrote as part of addressing comments.

**Rules:**
- New code you add should have tests
- Don't remove existing tests when fixing comments
- If a reviewer asks you to delete code, also delete its tests, but verify coverage hasn't dropped on remaining lines

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

## GitHub abuse rate limits

Two distinct rate limits exist — they have different causes and recovery times:

| Error | HTTP code | Cause | Recovery |
|---|---|---|---|
| `{"code":"abuse"}` | 403 | Secondary rate limit — too many write operations (comments, mutations) in a short window | Wait **2–3 minutes**. 60s is often not enough. |
| `{"message":"API rate limit exceeded"}` | 429 | Primary rate limit — too many API calls per hour | Wait until `X-RateLimit-Reset` header timestamp |

**Prevention:** Add `sleep 3` between individual thread reply API calls. When posting >20 replies, increase to `sleep 5`.

**Recovery from secondary rate limit (403):**
1. Stop all API writes immediately
2. Wait **2 minutes minimum** (not 60s — secondary limits are stricter)
3. Resume with `sleep 3` between each call
4. If 403 persists after 2 min, wait another 2 min before retrying

Never batch all replies in a tight loop — always space them out.

## Parallel thread resolution

When a PR has more than 10 unresolved threads, addressing one commit per thread is slow. Use this strategy instead:

### Group by file, batch per commit

1. Sort `ALL_THREADS` by `path` — threads in the same file can share a single commit.
2. Fix all threads in one file → `git commit` → `git push` → reply to **all** those threads with the same SHA → resolve them all.
3. Move to the next file group and repeat.

This reduces N commits to (number of files touched), which is usually 3–5 instead of 15–30.

### Posting replies concurrently (for large batches)

For truly independent thread groups (different files, no shared logic), you can post replies in parallel using background subshells — but always space out API writes:

```bash
# Post replies to a batch of threads concurrently, 3s apart
(
  sleep 3
  gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments/{ID1}/replies \
    -f body="🤖 Fixed in [${FULL_SHA:0:9}](https://github.com/Significant-Gravitas/AutoGPT/commit/${FULL_SHA}): ..."
) &
(
  sleep 6
  gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments/{ID2}/replies \
    -f body="🤖 Fixed in [${FULL_SHA:0:9}](https://github.com/Significant-Gravitas/AutoGPT/commit/${FULL_SHA}): ..."
) &
wait  # wait for all background replies before resolving
```

Then resolve sequentially (GraphQL mutations):
```bash
for THREAD_ID in "$THREAD1" "$THREAD2" "$THREAD3"; do
  gh api graphql -f query="mutation { resolveReviewThread(input: {threadId: \"${THREAD_ID}\"}) { thread { isResolved } } }"
  sleep 3
done
```

**Always sleep 3s between individual API writes** — GitHub's secondary rate limit (403) triggers on bursts of >20 writes. Increase to `sleep 5` when posting more than 20 replies in a batch.

## Resolving threads via GraphQL

Use `resolveReviewThread` **only after** the commit is pushed and the reply is posted:

```bash
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "THREAD_ID"}) { thread { isResolved } } }'
```

**Never call this mutation before committing the fix.** The orchestrator will verify actual unresolved counts via GraphQL after you output `ORCHESTRATOR:DONE` — false resolutions will be caught and you will be re-briefed.

### Verify actual count before outputting ORCHESTRATOR:DONE

Before claiming "0 unresolved threads", always query GitHub directly — don't rely on your own bookkeeping. Paginate all pages — a single `first: 100` query misses threads beyond page 1:

```bash
# Step 1: get total thread count
gh api graphql -f query='
{
  repository(owner: "Significant-Gravitas", name: "AutoGPT") {
    pullRequest(number: {N}) {
      reviewThreads { totalCount }
    }
  }
}' | jq '.data.repository.pullRequest.reviewThreads.totalCount'

# Step 2: paginate all pages, count truly unresolved
CURSOR=""; UNRESOLVED=0
while true; do
  AFTER=${CURSOR:+", after: \"$CURSOR\""}
  PAGE=$(gh api graphql -f query="
  {
    repository(owner: \"Significant-Gravitas\", name: \"AutoGPT\") {
      pullRequest(number: {N}) {
        reviewThreads(first: 100${AFTER}) {
          pageInfo { hasNextPage endCursor }
          nodes { isResolved }
        }
      }
    }
  }")
  UNRESOLVED=$(( UNRESOLVED + $(echo "$PAGE" | jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved==false)] | length') ))
  HAS_NEXT=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage')
  CURSOR=$(echo "$PAGE" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo.endCursor')
  [ "$HAS_NEXT" = "false" ] && break
done
echo "Unresolved threads: $UNRESOLVED"
```

Only output `ORCHESTRATOR:DONE` after this loop reports 0.
