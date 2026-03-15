---
name: babysit-pr
description: Address all open PR review comments, monitor CI, and loop until everything is clear. Fetches comments from all sources (reviewers, bots, sentry), addresses each one, reacts, replies, commits, pushes, and re-checks. TRIGGER when user asks to address review comments, fix PR feedback, respond to reviewer comments, or babysit/monitor a PR. Also triggered by "/address-comments".
user-invocable: true
metadata:
  author: autogpt-team
  version: "2.0.0"
---

# Babysit PR

Address all review comments and monitor CI until the PR is fully clear.

## Find the PR

```bash
gh pr list --head $(git branch --show-current) --repo Significant-Gravitas/AutoGPT
```

Or use the PR number/URL if provided by the user.

## Fetch comments (all sources)

Query all three GitHub comment APIs:

```bash
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews       # top-level reviews
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments      # inline review comments
gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments     # PR conversation comments
```

**Important bots to watch for:**
- `autogpt-reviewer` — posts "Blockers", "Should Fix", and "Nice to Have" sections. Address ALL of them, not just blockers.
- `sentry[bot]` — posts bug predictions. Evaluate each one — fix real bugs, explain false positives.
- `coderabbitai[bot]` — automated code review. Address actionable items.

## For each unreacted comment

1. Read the referenced code and understand the issue
2. Make the fix (or reply explaining why you disagree)
3. React and reply:

| Comment type | React API | Reply API |
|---|---|---|
| Inline review (`pulls/{N}/comments`) | `gh api repos/.../pulls/comments/{ID}/reactions -f content="+1"` | `gh api repos/.../pulls/{N}/comments/{ID}/replies -f body="..."` |
| PR conversation (`issues/{N}/comments`) | `gh api repos/.../issues/comments/{ID}/reactions -f content="+1"` | Post new issue comment |
| Top-level reviews | No reaction API | Address in code, reply via issue comment |

Skip comments already reacted to by the PR author.

## After addressing comments

1. Format: run `/check` (or manually: `poetry run format` for backend, `pnpm format` for frontend)
2. Commit with descriptive message
3. **Push immediately** — never batch commits without pushing
4. Re-fetch comments — address any new ones before waiting on CI

## While CI runs

Stay productive — don't idle:
1. Run pending local tests and fix failures
2. Address any remaining comments
3. Only poll `gh pr checks {N}` as last resort when nothing else to do

## The monitoring loop

```
address comments → format → commit → push
→ re-check comments → fix new ones → push
→ wait for CI → re-check comments after CI settles
→ repeat until: all comments reacted AND CI green AND no new comments arriving
```

**The loop ends when:** CI is fully green, all comments are addressed (reacted), and no new comments have appeared since CI settled. If CI triggers new bot reviews, the loop continues.

## Rules

- One todo item per comment for tracking
- React to every comment: `+1` = addressed, `-1` = disagreed (with explanation)
- Always push immediately after committing
- Reply on threads for inline comments; post new issue comment for conversation comments
- For backend commits in worktrees: `cd autogpt_platform/backend && poetry run git commit`
