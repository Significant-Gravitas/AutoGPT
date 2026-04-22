---
name: pr-polish
description: Alternate /pr-review and /pr-address on a PR until the PR is truly mergeable — no new review findings, zero unresolved inline threads, zero unaddressed top-level reviews or issue comments, all CI checks green, and two consecutive quiet polls after CI settles. Use when the user wants a PR polished to merge-ready without setting a fixed number of rounds.
user-invocable: true
argument-hint: "[PR number or URL] — if omitted, finds PR for current branch."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# PR Polish

**Goal.** Drive a PR to merge-ready by alternating `/pr-review` and `/pr-address` until **all** of the following hold:

1. The most recent `/pr-review` produces **zero new findings** (no new inline comments, no new top-level reviews with a non-empty body).
2. Every inline review thread reachable via GraphQL reports `isResolved: true`.
3. Every non-bot, non-author top-level review has been acknowledged (replied-to) OR resolved via a thread it spawned.
4. Every non-bot, non-author issue comment has been acknowledged (replied-to).
5. Every CI check is `conclusion: "success"` or `"skipped"` / `"neutral"` — none `"failure"` or still pending.
6. **Two consecutive post-CI polls** (≥60s apart) stay clean — no new threads, no new non-empty reviews, no new issue comments. Bots (coderabbitai, sentry, autogpt-reviewer) frequently post late after CI settles; a single green snapshot is not sufficient.

**Do not stop at a fixed number of rounds.** If round N introduces new comments, round N+1 is required. Cap at `_MAX_ROUNDS = 10` as a safety valve, but expect 2–5 in practice.

## TodoWrite

Before starting, write two todos so the user can see the loop progression:

- `Round {current}: /pr-review + /pr-address on PR #{N}` — current iteration.
- `Final polish polling: 2 consecutive clean polls, CI green, 0 unresolved` — runs after the last non-empty review round.

Update the `current` round counter at the start of each iteration; mark `completed` only when the round's address step finishes (all new threads addressed + resolved).

## Find the PR

```bash
ARG_PR="${ARG:-}"
# Normalize URL → numeric ID if the skill arg is a pull-request URL.
if [[ "$ARG_PR" =~ ^https?://github\.com/[^/]+/[^/]+/pull/([0-9]+) ]]; then
  ARG_PR="${BASH_REMATCH[1]}"
fi
PR="${ARG_PR:-$(gh pr list --head "$(git branch --show-current)" --repo Significant-Gravitas/AutoGPT --json number --jq '.[0].number')}"
if [ -z "$PR" ] || [ "$PR" = "null" ]; then
  echo "No PR found for current branch. Provide a PR number or URL as the skill arg."
  exit 1
fi
echo "Polishing PR #$PR"
```

## The outer loop

```text
round = 0
while round < _MAX_ROUNDS:
    round += 1
    baseline = snapshot_state(PR)   # see "Snapshotting state" below
    invoke_skill("pr-review", PR)   # posts findings as inline comments / top-level review
    findings = diff_state(PR, baseline)
    if findings.total == 0:
        break  # no new findings → go to polish polling
    invoke_skill("pr-address", PR)  # resolves every unresolved thread + CI failure
# Post-loop: polish polling (see below).
polish_polling(PR)
```

### Snapshotting state

Before each `/pr-review`, capture a baseline so the diff after the review reflects **only** what the review just added (not pre-existing threads):

```bash
# Inline threads — total count + latest databaseId per thread
gh api graphql -f query="
{
  repository(owner: \"Significant-Gravitas\", name: \"AutoGPT\") {
    pullRequest(number: ${PR}) {
      reviewThreads(first: 100) {
        totalCount
        nodes {
          id
          isResolved
          comments(last: 1) { nodes { databaseId } }
        }
      }
    }
  }
}" > /tmp/baseline_threads.json

# Top-level reviews — count + latest id per non-empty review
gh api "repos/Significant-Gravitas/AutoGPT/pulls/${PR}/reviews" --paginate \
  --jq '[.[] | select((.body // "") != "") | {id, user: .user.login, state, submitted_at}]' \
  > /tmp/baseline_reviews.json

# Issue comments — count + latest id per non-bot, non-author comment.
# Bots are filtered by User.type == "Bot" (GitHub sets this for app/bot
# accounts like coderabbitai, github-actions, sentry-io). The author is
# filtered by comparing login to the PR author — export it so jq can see it.
AUTHOR=$(gh api "repos/Significant-Gravitas/AutoGPT/pulls/${PR}" --jq '.user.login')
gh api "repos/Significant-Gravitas/AutoGPT/issues/${PR}/comments" --paginate \
  --jq --arg author "$AUTHOR" \
      '[.[] | select(.user.type != "Bot" and .user.login != $author)
            | {id, user: .user.login, created_at}]' \
  > /tmp/baseline_issue_comments.json
```

### Diffing after a review

After `/pr-review` runs, any of these counting as "new findings" means another address round is needed:

- New inline thread `id` not in the baseline.
- An existing thread whose latest comment `databaseId` is higher than the baseline's (new reply on an old thread).
- A new top-level review `id` with a non-empty body.
- A new issue comment `id` from a non-bot, non-author user.

If any of the four buckets is non-empty → not done; invoke `/pr-address` and loop.

## Polish polling

Once `/pr-review` produces zero new findings, do **not** exit yet. Bots (coderabbitai, sentry, autogpt-reviewer) commonly post late reviews after CI settles — 30–90 seconds after the final push. Poll at 60-second intervals:

```text
NON_SUCCESS_TERMINAL = {"failure", "cancelled", "timed_out", "action_required", "startup_failure"}
clean_polls = 0
required_clean = 2
while clean_polls < required_clean:
    # 1. CI gate — any terminal non-success conclusion (not just "failure")
    # must trigger /pr-address. "success", "skipped", "neutral" are clean;
    # anything else (including cancelled, timed_out, action_required) is a
    # blocker that won't self-resolve.
    ci = fetch_check_runs(PR)
    if any ci.conclusion in NON_SUCCESS_TERMINAL:
        invoke_skill("pr-address", PR)  # address failures + any new comments
        baseline = snapshot_state(PR)   # reset — push during address invalidates old baseline
        clean_polls = 0
        continue
    if any ci.conclusion is None (still in_progress):
        sleep 60; continue  # wait without counting this as clean

    # 2. Comment / thread gate
    threads = fetch_unresolved_threads(PR)
    new_issue_comments = diff_against_baseline(issue_comments)
    new_reviews = diff_against_baseline(reviews)
    if threads or new_issue_comments or new_reviews:
        invoke_skill("pr-address", PR)
        baseline = snapshot_state(PR)   # reset — the address loop just dealt with these,
                                        # otherwise they stay "new" relative to the old baseline forever
        clean_polls = 0
        continue

    # 3. Mergeability gate
    mergeable = gh api repos/.../pulls/${PR} --jq '.mergeable'
    if mergeable == false (CONFLICTING):
        resolve_conflicts(PR)  # see pr-address skill
        clean_polls = 0
        continue
    if mergeable is null (UNKNOWN):
        sleep 60; continue

    clean_polls += 1
    sleep 60
```

Only after `clean_polls == 2` do you report `ORCHESTRATOR:DONE`.

### Why 2 clean polls, not 1

A single green snapshot can be misleading — the final CI check often completes ~30s before a bot posts its delayed review. One quiet cycle does not prove the PR is stable; two consecutive cycles with no new threads, reviews, or issue comments arriving gives high confidence nothing else is incoming.

### Why checking every source each poll

`/pr-address` polling inside a single round already re-checks its own comments, but `/pr-polish` sits a level above and must also catch:

- New top-level reviews (autogpt-reviewer sometimes posts structured feedback only after several CI green cycles).
- Issue comments from human reviewers (not caught by inline thread polling).
- Sentry bug predictions that land on new line numbers post-push.
- Merge conflicts introduced by a race between your push and a merge to `dev`.

## Invocation pattern

Delegate to existing skills with the `Skill` tool; do not re-implement the review or address logic inline. This keeps the polish loop focused on orchestration and lets the child skills evolve independently.

```python
Skill(skill="pr-review",  args=pr_url)
Skill(skill="pr-address", args=pr_url)
```

After each child invocation, re-query GitHub state directly — never trust a summary for the stop condition. The orchestrator's `ORCHESTRATOR:DONE` is verified against actual GraphQL / REST responses per the rules in `pr-address`'s "Verify actual count before outputting ORCHESTRATOR:DONE" section.

### **Auto-continue: do NOT end your response between child skills**

`/pr-polish` is a single orchestration task — one invocation drives the PR all the way to merge-ready. When a child `Skill()` call returns control to you:

- Do NOT summarize and stop.
- Do NOT wait for user confirmation to continue.
- Immediately, in the same response, perform the next loop step: state diff → decide next action → next `Skill()` call or polling sleep.

The child skill returning is a **loop iteration boundary**, not a conversation turn boundary. You are expected to keep going until one of the exit conditions in the opening section is met (2 consecutive clean polls, `_MAX_ROUNDS` hit, or an unrecoverable error).

If the user needs to approve a risky action mid-loop (e.g., a force-push or a destructive git operation), pause there — but not at the routine "round N finished, round N+1 needed" boundary. Those are silent transitions.

## GitHub rate limits

This skill issues many GraphQL calls (one review-thread query per outer iteration plus per-poll queries inside polish polling). Expect the GraphQL budget to be tight on large PRs. When `gh api rate_limit --jq .resources.graphql.remaining` drops below ~200, back off:

- Fall back to REST for reads (flat `/pulls/{N}/comments`, `/pulls/{N}/reviews`, `/issues/{N}/comments`) per the `pr-address` skill's GraphQL-fallback section.
- Queue thread resolutions (GraphQL-only) until the budget resets; keep making progress on fixes + REST replies meanwhile.
- `sleep 5` between any batch of ≥20 writes to avoid secondary rate limits.

## Safety valves

- `_MAX_ROUNDS = 10` — if review+address rounds exceed this, stop and escalate to the user with a summary of what's still unresolved. A PR that cannot converge in 10 rounds has systemic issues that need human judgment.
- After each commit, run `poetry run format` / `pnpm format && pnpm lint && pnpm types` per the target codebase's conventions. A failing format check is CI `failure` that will never self-resolve.
- Every `/pr-review` round checks for **duplicate** concerns first (via `pr-review`'s own "Fetch existing review comments" step) so the loop does not re-post the same finding that a prior round already resolved.

## Reporting

When the skill finishes (either via two clean polls or hitting `_MAX_ROUNDS`), produce a compact summary:

```
PR #{N} polish complete ({rounds_completed} rounds):
- {X} inline threads opened and resolved
- {Y} CI failures fixed
- {Z} new commits pushed
Final state: CI green, {total} threads all resolved, mergeable.
```

If exiting via `_MAX_ROUNDS`, flag explicitly:

```
PR #{N} polish stopped at {_MAX_ROUNDS} rounds — NOT merge-ready:
- {N} threads still unresolved: {titles}
- CI status: {summary}
Needs human review.
```

## When to use this skill

Use when the user says any of:
- "polish this PR"
- "keep reviewing and addressing until it's mergeable"
- "loop /pr-review + /pr-address until done"
- "make sure the PR is actually merge-ready"

Do **not** use when:
- User wants just one review pass (→ `/pr-review`).
- User wants to address already-posted comments without further self-review (→ `/pr-address`).
- A fixed round count is explicitly requested (e.g., "do 3 rounds") — honour the count instead of converging.
