# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

AutoGPT Platform is a monorepo containing:

- **Backend** (`backend`): Python FastAPI server with async support
- **Frontend** (`frontend`): Next.js React application
- **Shared Libraries** (`autogpt_libs`): Common Python utilities

## Component Documentation

- **Backend**: See @backend/CLAUDE.md for backend-specific commands, architecture, and development tasks
- **Frontend**: See @frontend/CLAUDE.md for frontend-specific commands, architecture, and development patterns

## Key Concepts

1. **Agent Graphs**: Workflow definitions stored as JSON, executed by the backend
2. **Blocks**: Reusable components in `backend/backend/blocks/` that perform specific tasks
3. **Integrations**: OAuth and API connections stored per user
4. **Store**: Marketplace for sharing agent templates
5. **Virus Scanning**: ClamAV integration for file upload security

### Environment Configuration

#### Configuration Files

- **Backend**: `backend/.env.default` (defaults) → `backend/.env` (user overrides)
- **Frontend**: `frontend/.env.default` (defaults) → `frontend/.env` (user overrides)
- **Platform**: `.env.default` (Supabase/shared defaults) → `.env` (user overrides)

#### Docker Environment Loading Order

1. `.env.default` files provide base configuration (tracked in git)
2. `.env` files provide user-specific overrides (gitignored)
3. Docker Compose `environment:` sections provide service-specific overrides
4. Shell environment variables have highest precedence

#### Key Points

- All services use hardcoded defaults in docker-compose files (no `${VARIABLE}` substitutions)
- The `env_file` directive loads variables INTO containers at runtime
- Backend/Frontend services use YAML anchors for consistent configuration
- Supabase services (`db/docker/docker-compose.yml`) follow the same pattern

### Branching Strategy

- **`dev`** is the main development branch. All PRs should target `dev`.
- **`master`** is the production branch. Only used for production releases.

### Creating Pull Requests

- Create the PR against the `dev` branch of the repository.
- **Split PRs by concern** — each PR should have a single clear purpose. For example, "usage tracking" and "credit charging" should be separate PRs even if related. Combining multiple concerns makes it harder for reviewers to understand what belongs to what.
- Ensure the branch name is descriptive (e.g., `feature/add-new-block`)
- Use conventional commit messages (see below)
- **Structure the PR description with Why / What / How** — Why: the motivation (what problem it solves, what's broken/missing without it); What: high-level summary of changes; How: approach, key implementation details, or architecture decisions. Reviewers need all three to judge whether the approach fits the problem.
- Fill out the .github/PULL_REQUEST_TEMPLATE.md template as the PR description
- Always use `--body-file` to pass PR body — avoids shell interpretation of backticks and special characters:
  ```bash
  PR_BODY=$(mktemp)
  cat > "$PR_BODY" << 'PREOF'
  ## Summary
  - use `backticks` freely here
  PREOF
  gh pr create --title "..." --body-file "$PR_BODY" --base dev
  rm "$PR_BODY"
  ```
- Run the github pre-commit hooks to ensure code quality.

### Test-Driven Development (TDD)

When fixing a bug or adding a feature, follow a test-first approach:

1. **Write a failing test first** — create a test that reproduces the bug or validates the new behavior, marked with `@pytest.mark.xfail` (backend) or `.fixme` (Playwright). Run it to confirm it fails for the right reason.
2. **Implement the fix/feature** — write the minimal code to make the test pass.
3. **Remove the xfail marker** — once the test passes, remove the `xfail`/`.fixme` annotation and run the full test suite to confirm nothing else broke.

This ensures every change is covered by a test and that the test actually validates the intended behavior.

### Reviewing/Revising Pull Requests

Use `/pr-review` to review a PR or `/pr-address` to address comments.

When fetching comments manually:
- `gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/reviews --paginate` — top-level reviews
- `gh api repos/Significant-Gravitas/AutoGPT/pulls/{N}/comments --paginate` — inline review comments (always paginate to avoid missing comments beyond page 1)
- `gh api repos/Significant-Gravitas/AutoGPT/issues/{N}/comments` — PR conversation comments

### Conventional Commits

Use this format for commit messages and Pull Request titles:

**Conventional Commit Types:**

- `feat`: Introduces a new feature to the codebase
- `fix`: Patches a bug in the codebase
- `refactor`: Code change that neither fixes a bug nor adds a feature; also applies to removing features
- `ci`: Changes to CI configuration
- `docs`: Documentation-only changes
- `dx`: Improvements to the developer experience

**Recommended Base Scopes:**

- `platform`: Changes affecting both frontend and backend
- `frontend`
- `backend`
- `infra`
- `blocks`: Modifications/additions of individual blocks

**Subscope Examples:**

- `backend/executor`
- `backend/db`
- `frontend/builder` (includes changes to the block UI component)
- `infra/prod`

Use these scopes and subscopes for clarity and consistency in commit messages.
