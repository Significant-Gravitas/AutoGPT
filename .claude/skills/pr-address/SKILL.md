---
name: pr-address
description: Address all open PR review comments, fix CI failures, and loop until everything is clear. Fetches comments from all sources (reviewers, bots, sentry), fixes each one, reacts, replies, formats, commits, pushes, and re-checks. TRIGGER when user asks to address review comments, fix PR feedback, respond to reviewer comments, or monitor/babysit a PR.
user-invocable: true
args: "[PR number or URL] — optional, defaults to current branch's PR"
metadata:
  author: autogpt-team
  version: "3.0.0"
---

# PR Address

Address all review comments and monitor CI until the PR is fully clear.

## Find the PR

```bash
# From current branch
gh pr list --head $(git branch --show-current) --repo Significant-Gravitas/AutoGPT

# Or use the PR number/URL if provided as argument
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

## For each unreacted comment

1. Read the referenced code, make the fix (or reply explaining disagreement)
2. React and reply:

| Comment type | React | Reply |
|---|---|---|
| Inline (`pulls/{N}/comments`) | `gh api repos/.../pulls/comments/{ID}/reactions -f content="+1"` | `gh api repos/.../pulls/{N}/comments/{ID}/replies -f body="..."` |
| Conversation (`issues/{N}/comments`) | `gh api repos/.../issues/comments/{ID}/reactions -f content="+1"` | Post new issue comment |
| Top-level reviews | No reaction API | Address in code, reply via issue comment |

Skip comments already reacted to by the PR author.

## Format and commit

After fixing, format the changed code:

- **Backend** (from `autogpt_platform/backend/`): `poetry run format`
- **Frontend** (from `autogpt_platform/frontend/`): `pnpm format && pnpm lint && pnpm types`

Then commit and **push immediately** — never batch commits without pushing.

For backend commits in worktrees: `poetry run git commit` (pre-commit hooks).

## The loop

```
address comments → format → commit → push
→ re-check comments → fix new ones → push
→ wait for CI → re-check comments after CI settles
→ repeat until: all comments reacted AND CI green AND no new comments arriving
```

While CI runs, stay productive: run local tests, address remaining comments. Only poll CI as last resort.

**The loop ends when:** CI fully green + all comments addressed + no new comments since CI settled.

## Code quality rules (apply when fixing)

**Python:** top-level imports, no duck typing, Pydantic models, list comprehensions, early returns, lazy `%s` logging, `Security()` for auth deps, `transaction=True` for Redis pipelines, `max(0, val)` guards, `os.path.basename()` in error messages, `data:` for SSE events / `: comment` for heartbeats.

**Frontend:** function declarations (not arrows), no `useCallback`/`useMemo` unless needed, Tailwind only, no `dark:` classes, `<Link>` not `<a>`, Phosphor icons only, generated API hooks (not BackendAPI).

**Testing:** colocated `*_test.py`, mock where used not where defined, `AsyncMock` for async.
