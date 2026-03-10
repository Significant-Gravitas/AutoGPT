---
name: pr-create
description: Create a pull request for the current branch. TRIGGER when user asks to create a PR, open a pull request, push changes for review, or submit work for merging.
user-invocable: true
metadata:
  author: autogpt-team
  version: "1.0.0"
---

# Create Pull Request

## Steps

1. **Check for existing PR**: `gh pr view --json url -q .url 2>/dev/null` — if a PR already exists, output its URL and stop
2. **Understand changes**: `git status`, `git diff dev...HEAD`, `git log dev..HEAD --oneline`
3. **Read PR template**: `.github/PULL_REQUEST_TEMPLATE.md`
4. **Draft PR title**: Use conventional commits format (see CLAUDE.md for types and scopes)
5. **Fill out PR template** as the body — be thorough in the Changes section
6. **Format first** (if relevant changes exist):
   - Backend: `cd autogpt_platform/backend && poetry run format`
   - Frontend: `cd autogpt_platform/frontend && pnpm format`
   - Fix any lint errors, then commit formatting changes before pushing
7. **Push**: `git push -u origin HEAD`
8. **Create PR**: `gh pr create --base dev`
9. **Output** the PR URL

## Rules

- Always target `dev` branch
- Do NOT run tests — CI will handle that
- Use the PR template from `.github/PULL_REQUEST_TEMPLATE.md`
