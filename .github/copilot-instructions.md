# GitHub Copilot Instructions for AutoGPT

This file provides comprehensive onboarding information for GitHub Copilot coding agent to work efficiently with the AutoGPT repository.

## Repository Overview

**AutoGPT** is a powerful platform for creating, deploying, and managing continuous AI agents that automate complex workflows. This is a large monorepo (~150MB) containing multiple components:

- **AutoGPT Platform** (`autogpt_platform/`) - Main focus: Modern AI agent platform (Polyform Shield License)
- **Classic AutoGPT** (`classic/`) - Legacy agent system (MIT License)
- **Documentation** (`docs/`) - MkDocs-based documentation site
- **Infrastructure** - Docker configurations, CI/CD, and development tools

**Primary Languages & Frameworks:**
- **Backend**: Python 3.10-3.12, FastAPI, Prisma ORM, PostgreSQL, RabbitMQ
- **Frontend**: TypeScript, Next.js 15, React, Tailwind CSS, Radix UI
- **Development**: Docker, Poetry, pnpm, Playwright, Storybook

## Build and Validation Instructions

### Essential Setup Commands

**Always run these commands in the correct directory and in this order:**

1. **Initial Setup** (required once):
   ```bash
   # Clone and enter repository
   git clone <repo> && cd AutoGPT
   
   # Start all services (database, redis, rabbitmq, clamav)
   cd autogpt_platform && docker compose up -d
   ```

2. **Backend Setup** (always run before backend development):
   ```bash
   cd autogpt_platform/backend
   poetry install                    # Install dependencies
   poetry run prisma migrate dev     # Run database migrations
   poetry run prisma generate        # Generate Prisma client
   ```

3. **Frontend Setup** (always run before frontend development):
   ```bash
   cd autogpt_platform/frontend
   pnpm install                      # Install dependencies
   ```

### Runtime Requirements

**Critical:** Always ensure Docker services are running before starting development:
```bash
cd autogpt_platform && docker compose up -d
```

**Python Version:** Use Python 3.10-3.12 (Poetry manages this via pyproject.toml)
**Node.js Version:** Use Node.js 21+ with pnpm package manager

### Development Commands

**Backend Development:**
```bash
cd autogpt_platform/backend
poetry run serve                     # Start development server (port 8000)
poetry run test                      # Run all tests (requires ~5 minutes)
poetry run pytest path/to/test.py    # Run specific test
poetry run format                    # Format code (Black + isort) - always run first
poetry run lint                      # Lint code (ruff) - run after format
```

**Frontend Development:**
```bash
cd autogpt_platform/frontend
pnpm dev                            # Start development server (port 3000)
pnpm build                          # Build for production
pnpm test                           # Run Playwright E2E tests (requires build first)
pnpm test-ui                        # Run tests with UI
pnpm format                         # Format and lint code
pnpm storybook                      # Start component development server
```

### Testing Strategy

**Backend Tests:**
- **Block Tests**: `poetry run pytest backend/blocks/test/test_block.py -xvs` (validates all blocks)
- **Specific Block**: `poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[BlockName]' -xvs`
- **Snapshot Tests**: Use `--snapshot-update` when output changes, always review with `git diff`

**Frontend Tests:**
- **E2E Tests**: Always run `pnpm build` before `pnpm test` (Playwright requires build)
- **Component Tests**: Use Storybook for isolated component development

### Critical Validation Steps

**Before committing changes:**
1. Run `poetry run format` (backend) and `pnpm format` (frontend)
2. Ensure all tests pass in modified areas
3. Verify Docker services are still running
4. Check that database migrations apply cleanly

**Common Issues & Workarounds:**
- **Prisma issues**: Run `poetry run prisma generate` after schema changes
- **Permission errors**: Ensure Docker has proper permissions
- **Port conflicts**: Check ports 3000 (frontend), 8000 (backend), 5432 (postgres)
- **Test timeouts**: Backend tests can take 5+ minutes, use `-x` flag to stop on first failure

## Project Layout & Architecture

### Core Architecture

**AutoGPT Platform** (`autogpt_platform/`):
- `backend/` - FastAPI server with async support
  - `backend/backend/` - Core API logic
  - `backend/blocks/` - Agent execution blocks
  - `backend/data/` - Database models and schemas
  - `schema.prisma` - Database schema definition
- `frontend/` - Next.js application
  - `src/app/` - App Router pages and layouts
  - `src/components/` - Reusable React components
  - `src/lib/` - Utilities and configurations
- `autogpt_libs/` - Shared Python utilities
- `docker-compose.yml` - Development stack orchestration

**Key Configuration Files:**
- `pyproject.toml` - Python dependencies and tooling
- `package.json` - Node.js dependencies and scripts
- `schema.prisma` - Database schema and migrations
- `next.config.mjs` - Next.js configuration
- `tailwind.config.ts` - Styling configuration

### Security & Middleware

**Cache Protection**: Backend includes middleware preventing sensitive data caching in browsers/proxies
**Authentication**: JWT-based with Supabase integration
**User ID Validation**: All data access requires user ID checks - verify this for any `data/*.py` changes

### Development Workflow

**GitHub Actions**: Multiple CI/CD workflows in `.github/workflows/`
- `platform-backend-ci.yml` - Backend testing and validation
- `platform-frontend-ci.yml` - Frontend testing and validation
- `platform-fullstack-ci.yml` - End-to-end integration tests

**Pre-commit Hooks**: Run linting and formatting checks
**Conventional Commits**: Use format `type(scope): description` (e.g., `feat(backend): add API`)

### Key Source Files

**Backend Entry Points:**
- `backend/backend/server/server.py` - FastAPI application setup
- `backend/backend/data/` - Database models and user management
- `backend/blocks/` - Agent execution blocks and logic

**Frontend Entry Points:**
- `frontend/src/app/layout.tsx` - Root application layout
- `frontend/src/app/page.tsx` - Home page
- `frontend/src/lib/supabase/` - Authentication and database client

**Protected Routes**: Update `frontend/lib/supabase/middleware.ts` when adding protected routes

### Agent Block System

Agents are built using a visual block-based system where each block performs a single action. Blocks are defined in `backend/blocks/` and must include:
- Block definition with input/output schemas
- Execution logic with proper error handling
- Tests validating functionality

### Database & ORM

**Prisma ORM** with PostgreSQL backend including pgvector for embeddings:
- Schema in `schema.prisma`
- Migrations in `backend/migrations/`
- Always run `prisma migrate dev` and `prisma generate` after schema changes

## Trust These Instructions

These instructions are comprehensive and tested. Only perform additional searches if:
1. Information here is incomplete for your specific task
2. You encounter errors not covered by the workarounds
3. You need to understand implementation details not covered above

For detailed platform development patterns, refer to `autogpt_platform/CLAUDE.md` and `AGENTS.md` in the repository root.