---
name: pr-review
description: Review a pull request for code quality, bugs, security, and architecture. Reads the diff, analyzes changes, and provides structured feedback. TRIGGER when user asks to review a PR, review code changes, or do a code review. NOT for addressing existing comments (use /pr-address for that).
user-invocable: true
args: "[PR number or URL] — optional, defaults to current branch's PR"
metadata:
  author: autogpt-team
  version: "2.0.0"
---

# PR Review

Review a pull request and provide structured feedback.

## Get the PR

```bash
gh pr diff {N}       # or: gh pr diff (current branch)
gh pr view {N}       # read description for context
```

## Review checklist

For each changed file, evaluate:

### Correctness
- Logic errors, off-by-one, missing edge cases
- Race conditions (TOCTOU in file access, credit charging)
- Error handling — exceptions caught at the right level?
- Async correctness — missing `await`, unclosed resources

### Security
- Input validation at system boundaries
- No command injection, XSS, SQL injection
- Secrets not hardcoded or logged
- File path sanitization (`os.path.basename()` in error messages)

### Code quality

**Python:** top-level imports, no duck typing, Pydantic models, list comprehensions, early returns, lazy `%s` logging (not f-strings in log calls), no linter suppressors (`# type: ignore`, `# noqa`), `Security()` for FastAPI auth deps, `transaction=True` for Redis pipelines, `max(0, val)` guards, `data:` for SSE events / `: comment` for heartbeats.

**Frontend:** function declarations (not arrows for components), no unnecessary `useCallback`/`useMemo`, Tailwind only, no `dark:` classes, `<Link>` not `<a>`, Phosphor icons only, generated API hooks (not BackendAPI), no `any` types, capitalize acronyms (`graphID`).

### Architecture
- DRY — duplicated logic that should be extracted
- Single responsibility — functions doing too much
- Modular — can functions be split smaller?

### Testing
- Edge cases tested?
- Colocated `*_test.py` (backend), `__tests__/` (frontend)
- Mocks target where symbol is **used**, not where defined
- Mock at boundaries, not internals

## Output format

Three tiers:
1. **Blockers** — must fix before merge (bugs, security, broken functionality)
2. **Should Fix** — important improvements (quality, missing tests, races)
3. **Nice to Have** — minor suggestions (naming, style)

For each item: file path, line number, description, suggested fix.

## Rules

- Read the actual code, not just the diff — context matters
- Check mock targets match module paths after refactoring
- If PR touches SSE/streaming, verify the protocol format
