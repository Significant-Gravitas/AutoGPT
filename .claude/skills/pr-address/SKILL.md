---
name: pr-address
description: Address PR review comments and loop until CI green and all comments resolved. TRIGGER when user asks to address comments, fix PR feedback, respond to reviewers, or babysit/monitor a PR.
user-invocable: true
args: "[PR number or URL] — if omitted, finds PR for current branch."
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

```bash
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews       # top-level reviews
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments      # inline review comments
gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments     # PR conversation comments
```

**Bots to watch for:**
- `autogpt-reviewer` — posts "Blockers", "Should Fix", "Nice to Have". Address ALL of them.
- `sentry[bot]` — bug predictions. Fix real bugs, explain false positives.
- `coderabbitai[bot]` — automated review. Address actionable items.

## For each unaddressed comment

Address comments **one at a time**: fix → commit → push → inline reply → next.

1. Read the referenced code, make the fix (or reply explaining why it's not needed)
2. Commit and push the fix
3. Reply **inline** (not as a new top-level comment) referencing the fixing commit — this is what resolves the conversation for bot reviewers (coderabbitai, sentry):

| Comment type | How to reply |
|---|---|
| Inline review (`pulls/{N}/comments`) | `gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments/{ID}/replies -f body="Fixed in <commit-sha>: <description>"` |
| Conversation (`issues/{N}/comments`) | `gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments -f body="Fixed in <commit-sha>: <description>"` |

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

### Waiting for CI + new comments

Wait for **both** CI completion and new PR comments simultaneously. Do not block only on CI — bots often post new comments on fresh commits while CI is still running.

**Combined wait approach:**

1. Record current comment count before waiting:
```bash
COMMENT_COUNT=$(( \
  $(gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments --jq 'length') + \
  $(gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments --jq 'length') + \
  $(gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews --jq 'length') \
))
```

2. Start CI watch in background:
```bash
gh pr checks {N} --repo Significant-Gravitas/AutoGPT --watch --fail-fast &
CI_PID=$!
```

3. Poll for new comments every 30 seconds while CI runs:
```bash
while kill -0 $CI_PID 2>/dev/null; do
  sleep 30
  c1=$(gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments --jq 'length') || continue
  c2=$(gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments --jq 'length') || continue
  c3=$(gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews --jq 'length') || continue
  NEW_COUNT=$((c1 + c2 + c3))
  if [ "$NEW_COUNT" -gt "$COMMENT_COUNT" ]; then
    echo "New comments detected ($COMMENT_COUNT → $NEW_COUNT)"
    break
  fi
done
```

4. When new comments arrive, address them immediately while CI continues in the background.
   If you push new commits, the old CI_PID becomes stale (new commits trigger new CI runs) — restart the combined wait from step 1 (recompute COMMENT_COUNT and start a fresh CI watch for the new HEAD).
   Otherwise, update COMMENT_COUNT and resume polling.

5. When CI finishes (CI_PID exits), collect its exit status:
```bash
wait $CI_PID
CI_EXIT=$?
```

If CI failed:
1. Get failed check links: `gh pr checks {N} --repo Significant-Gravitas/AutoGPT --json bucket,link --jq '.[] | select(.bucket == "fail") | .link'`
2. Extract the run ID from the link (format: `.../actions/runs/<run-id>/job/...`), then view logs: `gh run view <run-id> --log-failed`
3. Fix → commit → push → restart the combined wait (from step 1)

**The loop ends when:** CI fully green + all comments addressed + no new comments since CI settled.
