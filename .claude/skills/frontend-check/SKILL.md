---
name: frontend-check
description: Run the full frontend formatting, linting, and type checking suite. Ensures code quality before commits and PRs. TRIGGER when frontend TypeScript/React code has been modified and needs validation.
user-invocable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Frontend Check

## Steps (in order)

1. **Format**: `pnpm format` — NEVER run individual formatters
2. **Lint**: `pnpm lint` — fix errors, re-run until clean
3. **Types**: `pnpm types` — if it keeps failing after multiple attempts, stop and ask the user
