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
- Ensure the branch name is descriptive (e.g., `feature/add-new-block`)
- Use conventional commit messages (see below)
- Fill out the .github/PULL_REQUEST_TEMPLATE.md template as the PR description
- Run the github pre-commit hooks to ensure code quality.

### Reviewing/Revising Pull Requests

- When the user runs /pr-comments or tries to fetch them, also run gh api /repos/Significant-Gravitas/AutoGPT/pulls/[issuenum]/reviews to get the reviews
- Use gh api /repos/Significant-Gravitas/AutoGPT/pulls/[issuenum]/reviews/[review_id]/comments to get the review contents
- Use gh api /repos/Significant-Gravitas/AutoGPT/issues/9924/comments to get the pr specific comments

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
