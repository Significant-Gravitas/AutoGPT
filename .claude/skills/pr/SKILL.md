---
name: pr
description: Unified PR skill — review a PR for quality issues OR address existing review comments. Auto-detects mode based on context, or specify with args. TRIGGER when user asks to review a PR, address review comments, fix PR feedback, respond to reviewers, or babysit/monitor a PR.
user-invocable: true
args: "[review|address] [PR number or URL] — 'review' reads diff and gives feedback, 'address' fixes comments and loops until CI green. If omitted, auto-detects from context."
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# PR

## Mode detection

- **`/pr review`** or `/pr review 12345` — review a PR for quality issues
- **`/pr address`** or `/pr address 12345` — address existing comments and loop until clear
- **`/pr 12345`** — auto-detect: if there are unreacted comments → address mode, otherwise → review mode
- **`/pr`** — find PR for current branch, then auto-detect

## Find the PR

```bash
gh pr list --head $(git branch --show-current) --repo Significant-Gravitas/AutoGPT
gh pr view {N}
```

---

## Review mode

Read the diff and provide structured feedback:

```bash
gh pr diff {N}
```

### What to check

**Correctness:** logic errors, off-by-one, missing edge cases, race conditions (TOCTOU in file access, credit charging), error handling gaps, async correctness (missing `await`, unclosed resources).

**Security:** input validation at boundaries, no injection (command, XSS, SQL), secrets not logged, file paths sanitized (`os.path.basename()` in error messages).

**Code quality:** see Code Quality Rules below.

**Architecture:** DRY, single responsibility, modular functions. `Security()` vs `Depends()` for FastAPI auth. `data:` for SSE events, `: comment` for heartbeats. `transaction=True` for Redis pipelines.

**Testing:** edge cases covered, colocated `*_test.py` (backend) / `__tests__/` (frontend), mocks target where symbol is **used** not defined, `AsyncMock` for async.

### Output format

Three tiers:
1. **Blockers** — must fix before merge
2. **Should Fix** — important improvements
3. **Nice to Have** — minor suggestions

For each: file path, line number, description, suggested fix.

---

## Address mode

Fix all review comments and loop until CI is green.

### Fetch comments (all sources)

```bash
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews       # top-level reviews
gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments      # inline review comments
gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments     # PR conversation comments
```

**Bots to watch for:**
- `autogpt-reviewer` — posts "Blockers", "Should Fix", "Nice to Have". Address ALL of them.
- `sentry[bot]` — bug predictions. Fix real bugs, explain false positives.
- `coderabbitai[bot]` — automated review. Address actionable items.

### For each unreacted comment

1. Read the referenced code, make the fix (or reply explaining disagreement)
2. React and reply:

| Comment type | React | Reply |
|---|---|---|
| Inline (`pulls/{N}/comments`) | `gh api repos/.../pulls/comments/{ID}/reactions -f content="+1"` | `gh api repos/.../pulls/{N}/comments/{ID}/replies -f body="..."` |
| Conversation (`issues/{N}/comments`) | `gh api repos/.../issues/comments/{ID}/reactions -f content="+1"` | Post new issue comment |
| Top-level reviews | No reaction API | Address in code, reply via issue comment |

### Format and commit

After fixing, format the changed code:

- **Backend** (from `autogpt_platform/backend/`): `poetry run format`
- **Frontend** (from `autogpt_platform/frontend/`): `pnpm format && pnpm lint && pnpm types`

If API routes changed, regenerate the frontend client (run in one shell block):
```bash
cd autogpt_platform/backend && poetry run rest &
REST_PID=$!
WAIT=0; until curl -sf http://localhost:8006/health > /dev/null 2>&1; do sleep 1; WAIT=$((WAIT+1)); [ $WAIT -ge 60 ] && echo "Timed out" && kill $REST_PID && exit 1; done
cd ../frontend && pnpm generate:api:force
kill $REST_PID
```
Never manually edit files in `src/app/api/__generated__/`.

Then commit and **push immediately** — never batch commits without pushing.

For backend commits in worktrees: `poetry run git commit` (pre-commit hooks).

### The loop

```text
address comments → format → commit → push
→ re-check comments → fix new ones → push
→ wait for CI → re-check comments after CI settles
→ repeat until: all comments reacted AND CI green AND no new comments arriving
```

While CI runs, stay productive: run local tests, address remaining comments.

**The loop ends when:** CI fully green + all comments addressed + no new comments since CI settled.

---

## Code quality rules (both modes)

**Python:** top-level imports, no duck typing, Pydantic models, list comprehensions, early returns, lazy `%s` logging, no linter suppressors, `Security()` for auth deps, `transaction=True` for Redis pipelines, `max(0, val)` guards, `os.path.basename()` in error messages, `data:` for SSE events / `: comment` for heartbeats.

**Frontend:** function declarations (not arrows), no `useCallback`/`useMemo` unless needed, Tailwind only, no `dark:` classes, `<Link>` not `<a>`, Phosphor icons only, generated API hooks (not BackendAPI), no `any` types. Hooks follow `use{Method}{Version}{OperationName}`.

**Testing:** colocated `*_test.py`, mock where used not where defined, `AsyncMock` for async.
