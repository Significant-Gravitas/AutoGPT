---
name: pr-create
description: Create a pull request for the current branch. Runs /check, pushes, and creates the PR with the repo template. TRIGGER when user asks to create a PR, open a pull request, push changes for review, or submit work for merging.
user-invocable: true
metadata:
  author: autogpt-team
  version: "2.0.0"
---

# Create Pull Request

## Steps

1. **Check for existing PR**: `gh pr view --json url -q .url 2>/dev/null` — if exists, output URL and stop
2. **Understand changes**: `git diff dev...HEAD`, `git log dev..HEAD --oneline`
3. **Run /check** — format and lint all modified code
4. **Read PR template**: `.github/PULL_REQUEST_TEMPLATE.md`
5. **Draft PR**:
   - Title: conventional commits format (`feat(scope)`, `fix(scope)`, etc.)
   - Body: fill out the PR template thoroughly
6. **Push**: `git push -u origin HEAD`
7. **Create**: `gh pr create --base dev`
8. **Output** the PR URL

## Rules

- Always target `dev` branch
- Do NOT run tests — CI handles that
- Use the PR template from `.github/PULL_REQUEST_TEMPLATE.md`
- For backend worktrees: `poetry run git commit` for pre-commit hooks
