---
name: pr-review
description: Review a PR for correctness, security, code quality, and testing issues. TRIGGER when user asks to review a PR, check PR quality, or give feedback on a PR.
user-invocable: true
args: "[PR number or URL] — if omitted, finds PR for current branch."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# PR Review

## Find the PR

```bash
gh pr list --head $(git branch --show-current) --repo Significant-Gravitas/AutoGPT
gh pr view {N}
```

## Read the diff

```bash
gh pr diff {N}
```

## Fetch existing review comments

Before posting anything, fetch existing inline comments to avoid duplicates:

```bash
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments --paginate
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews
```

## What to check

**Correctness:** logic errors, off-by-one, missing edge cases, race conditions (TOCTOU in file access, credit charging), error handling gaps, async correctness (missing `await`, unclosed resources).

**Security:** input validation at boundaries, no injection (command, XSS, SQL), secrets not logged, file paths sanitized (`os.path.basename()` in error messages).

**Code quality:** apply rules from backend/frontend CLAUDE.md files.

**Architecture:** DRY, single responsibility, modular functions. `Security()` vs `Depends()` for FastAPI auth. `data:` for SSE events, `: comment` for heartbeats. `transaction=True` for Redis pipelines.

**Testing:** edge cases covered, colocated `*_test.py` (backend) / `__tests__/` (frontend), mocks target where symbol is **used** not defined, `AsyncMock` for async.

## Output format

Every comment **must** be prefixed with `🤖` and a criticality badge:

| Tier | Badge | Meaning |
|---|---|---|
| Blocker | `🔴 **Blocker**` | Must fix before merge |
| Should Fix | `🟠 **Should Fix**` | Important improvement |
| Nice to Have | `🟡 **Nice to Have**` | Minor suggestion |
| Nit | `🔵 **Nit**` | Style / wording |

Example: `🤖 🔴 **Blocker**: Missing error handling for X — suggest wrapping in try/except.`

## Post inline comments

For each finding, post an inline comment on the PR (do not just write a local report):

```bash
# Get the latest commit SHA for the PR
COMMIT_SHA=$(gh api repos/Significant-Gravitas/AutoGPT/pulls/{N} --jq '.head.sha')

# Post an inline comment on a specific file/line
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments \
  -f body="🤖 🔴 **Blocker**: <description>" \
  -f commit_id="$COMMIT_SHA" \
  -f path="<file path>" \
  -F line=<line number>
```
