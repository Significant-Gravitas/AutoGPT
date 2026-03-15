---
name: check
description: Run formatting, linting, type checking, and tests for modified code. Auto-detects whether backend (Python) or frontend (TypeScript/React) code changed and runs the appropriate checks. TRIGGER after modifying code, before commits, before PRs, or when user asks to run checks/tests/lint/format. Replaces /backend-check and /frontend-check.
user-invocable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Check

Run code quality checks for modified code. Auto-detects backend vs frontend changes.

## Detect what changed

```bash
git diff --name-only HEAD
git diff --name-only --cached
```

- Files under `autogpt_platform/backend/` → run backend checks
- Files under `autogpt_platform/frontend/` → run frontend checks
- Both → run both

If unsure (e.g., user just says "run checks"), run both.

## Backend checks (Python)

All commands from `autogpt_platform/backend/`:

1. **Format + lint**: `poetry run format` — runs Black, isort, and ruff. NEVER run them individually.
2. Fix any remaining errors manually, re-run until clean.
3. **Test**: `poetry run test` (runs DB setup + pytest). For specific files: `poetry run pytest -s -vvv <test_files>`
4. **Snapshots** (if needed): `poetry run pytest path/to/test.py --snapshot-update` — review with `git diff`

## Frontend checks (TypeScript/React)

All commands from `autogpt_platform/frontend/`:

1. **Format**: `pnpm format` — NEVER run individual formatters
2. **Lint**: `pnpm lint` — fix errors, re-run until clean
3. **Types**: `pnpm types` — if it keeps failing after multiple attempts, stop and ask the user

## Shared code quality rules

Apply these regardless of backend or frontend:

- No linter suppressors (`# type: ignore`, `# noqa`, `// @ts-ignore`, `// eslint-disable`) — fix the actual issue
- No `any` types unless the value genuinely can be anything
- Remove unused imports, variables, and dead code
- Lazy logging: use `logger.info("msg %s", var)` not `logger.info(f"msg {var}")`
- Flag `dark:` Tailwind classes — the design system handles dark mode
- Flag `<a>` tags that should be Next.js `<Link>`

## Rules

- Always run format BEFORE lint — formatting fixes most lint issues
- For backend commits in worktrees: `poetry run git commit` (pre-commit hooks)
- If types keep failing, stop and ask — don't loop forever
