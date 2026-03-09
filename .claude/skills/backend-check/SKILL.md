---
name: backend-check
description: Run the full backend formatting, linting, and test suite. Ensures code quality before commits and PRs. TRIGGER when backend Python code has been modified and needs validation.
user-invocable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Backend Check

## Steps

1. **Format**: `poetry run format` — runs formatting AND linting. NEVER run ruff/black/isort individually
2. **Fix** any remaining errors manually, re-run until clean
3. **Test**: `poetry run test` (runs DB setup + pytest). For specific files: `poetry run pytest -s -vvv <test_files>`
4. **Snapshots** (if needed): `poetry run pytest path/to/test.py --snapshot-update` — review with `git diff`
