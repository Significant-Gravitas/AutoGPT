---
name: pr-review
description: Review a pull request for code quality, bugs, security, and architecture. Reads the diff, analyzes changes, and provides structured feedback. TRIGGER when user asks to review a PR, review code changes, or do a code review. NOT for addressing existing comments (use /babysit-pr for that).
user-invocable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# PR Review

Review a pull request and provide structured feedback on code quality, bugs, and architecture.

## Get the PR diff

```bash
# If PR number provided:
gh pr diff {N}

# If on the branch:
gh pr diff
```

Also read the PR description for context:
```bash
gh pr view {N}
```

## Review checklist

For each changed file, evaluate:

### Correctness
- Logic errors, off-by-one, missing edge cases
- Race conditions (TOCTOU in file access, credit charging, etc.)
- Error handling gaps — are exceptions caught at the right level?
- Async correctness — missing `await`, unclosed resources

### Security
- Input validation at system boundaries (user input, external APIs)
- No command injection, XSS, SQL injection
- Secrets not hardcoded or logged
- File path sanitization (use `os.path.basename()` for error messages)

### Code quality (apply /code-style rules)
- Python: top-level imports, no duck typing, Pydantic models, list comprehensions, early returns
- Frontend: function declarations, no unnecessary `useCallback`/`useMemo`, Tailwind only, Phosphor icons only
- No linter suppressors (`# type: ignore`, `# noqa`, `// @ts-ignore`)
- Lazy logging with `%s` format (not f-strings in log calls)

### Architecture
- DRY — duplicated logic that should be extracted
- Single responsibility — functions doing too much
- API design — `Security()` vs `Depends()` for auth in FastAPI OpenAPI spec
- SSE protocol — `data:` lines for parsed events, `: comment` lines for heartbeats/status
- Redis operations — use `transaction=True` for pipeline atomicity

### Testing
- Are edge cases tested?
- Test files colocated (`*_test.py` for backend, `__tests__/` for frontend)
- Mocks target the right module path (where the symbol is used, not where defined)
- No mocking of internals — mock at boundaries

## Output format

Provide feedback in three tiers:
1. **Blockers** — Must fix before merge (bugs, security issues, broken functionality)
2. **Should Fix** — Important improvements (code quality, missing tests, race conditions)
3. **Nice to Have** — Minor suggestions (naming, style, documentation)

For each item: file path, line number, description, and suggested fix.

## Rules

- Read the actual code, not just the diff — context matters
- Check that test mocks match the actual module paths after refactoring
- Flag `dark:` CSS classes if the design system handles dark mode
- Flag `<a>` tags that should be `<Link>` for Next.js routing
- If the PR touches SSE/streaming code, verify the protocol format
