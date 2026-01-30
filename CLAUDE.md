# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

AutoGPT is a monorepo containing two distinct systems under different licenses:

- **AutoGPT Platform** (`/autogpt_platform/`): The primary, actively developed product -- a graph-based workflow automation platform for building, deploying, and managing AI agents. Licensed under Polyform Shield License.
- **Classic AutoGPT** (`/classic/`): The original autonomous AI agent (CLI-based) and supporting tooling. Licensed under MIT.
- **Documentation** (`/docs/`): MkDocs-based documentation site.

Most development work happens in `autogpt_platform/`. See `/autogpt_platform/CLAUDE.md` for platform-specific guidance.

## Monorepo Structure

```
AutoGPT/
├── autogpt_platform/           # Main platform (Polyform Shield License)
│   ├── backend/                # Python FastAPI backend
│   ├── frontend/               # Next.js React frontend
│   ├── autogpt_libs/           # Shared Python libraries
│   ├── db/                     # Supabase/PostgreSQL Docker config
│   ├── graph_templates/        # Predefined agent workflow templates
│   ├── installer/              # Installation scripts
│   ├── docker-compose.yml      # Service orchestration
│   └── docker-compose.platform.yml
├── classic/                    # Classic AutoGPT (MIT License)
│   ├── original_autogpt/       # Original CLI-based agent
│   ├── forge/                  # Agent development framework
│   ├── benchmark/              # Agent benchmarking (agbenchmark)
│   └── frontend/               # Flutter desktop UI
├── docs/                       # MkDocs documentation site
├── .github/                    # CI/CD workflows, PR templates
└── .pre-commit-config.yaml     # Pre-commit hooks
```

## Tech Stack Summary

### Platform Backend (`autogpt_platform/backend/`)
- **Language:** Python 3.10-3.12
- **Framework:** FastAPI 0.116+ with Uvicorn
- **ORM:** Prisma (async Python client)
- **Database:** PostgreSQL with pgvector
- **Queue:** RabbitMQ (aio-pika)
- **Cache:** Redis
- **Auth:** Supabase + JWT
- **Package Manager:** Poetry
- **Testing:** pytest, pytest-asyncio, pytest-snapshot
- **Linting/Formatting:** Black, isort, Ruff, Flake8, Pyright

### Platform Frontend (`autogpt_platform/frontend/`)
- **Framework:** Next.js 15 (App Router)
- **Language:** TypeScript 5.9
- **UI Library:** React 18
- **Styling:** Tailwind CSS 3.4
- **Components:** Radix UI primitives
- **State Management:** React hooks, TanStack Query, Supabase client
- **Workflow Editor:** @xyflow/react
- **Package Manager:** pnpm 10.11+
- **Testing:** Playwright (E2E), Storybook (component)
- **Linting/Formatting:** ESLint, Prettier (with Tailwind plugin)
- **API Client:** Auto-generated via Orval from OpenAPI spec

### Classic Agent (`classic/original_autogpt/`)
- **Language:** Python 3.10+
- **CLI:** Click
- **Framework:** Forge (custom agent framework)
- **Package Manager:** Poetry

## Essential Commands

### Platform Backend

```bash
cd autogpt_platform/backend

# Install dependencies
poetry install

# Run database migrations
poetry run prisma migrate dev

# Generate Prisma client
poetry run prisma generate

# Start all infrastructure (PostgreSQL, Redis, RabbitMQ, ClamAV)
docker compose -f ../docker-compose.yml up -d

# Run the backend (all services)
poetry run app

# Run individual services
poetry run rest        # REST API (port 8006)
poetry run ws          # WebSocket server (port 8001)
poetry run executor    # Execution engine
poetry run scheduler   # Task scheduler
poetry run notification # Notification service

# Run all tests
poetry run test

# Run specific test file
poetry run pytest path/to/test_file.py::test_function_name

# Run block tests (validates all ~60 block types)
poetry run pytest backend/blocks/test/test_block.py -xvs

# Run test for specific block
poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[GetCurrentTimeBlock]' -xvs

# Format code (Black + isort)
poetry run format

# Lint code (Ruff)
poetry run lint

# Update test snapshots
poetry run pytest path/to/test.py --snapshot-update
```

### Platform Frontend

```bash
cd autogpt_platform/frontend

# Install dependencies
pnpm install

# Start dev server (with Turbo, auto-generates API client)
pnpm dev

# Build for production
pnpm build

# Run E2E tests (builds first)
pnpm test

# Run E2E tests without building
pnpm test:no-build

# Interactive test UI
pnpm test-ui

# Lint (ESLint + Prettier check)
pnpm lint

# Format (ESLint fix + Prettier write)
pnpm format

# Type check
pnpm types

# Storybook (component development, port 6006)
pnpm storybook

# Regenerate API client from backend OpenAPI spec
pnpm generate:api
```

### Docker Services

```bash
cd autogpt_platform

# Start all services
docker compose up -d

# Start with extended platform services (includes Supabase)
docker compose -f docker-compose.yml -f docker-compose.platform.yml up -d

# Scale executor service
docker compose up -d --scale executor=3

# View logs
docker compose logs -f rest_server executor

# Rebuild and restart specific service
docker compose build rest_server && docker compose up -d --no-deps rest_server
```

### Classic AutoGPT

```bash
cd classic/original_autogpt

# Install dependencies
poetry install

# Run the agent
poetry run autogpt run

# Run the agent protocol server
poetry run autogpt serve

# Run tests
poetry run pytest
```

## Architecture Overview

### Platform Backend Services

The platform runs as multiple microservices:

| Service | Port | Purpose |
|---------|------|---------|
| `rest_server` | 8006 | FastAPI REST API |
| `websocket_server` | 8001 | WebSocket API for real-time updates |
| `executor` | 8002 | Agent graph execution engine |
| `scheduler_server` | 8003 | Cron-triggered execution scheduling |
| `notification_server` | 8007 | Email/webhook notifications |
| `database_manager` | 8005 | Database connection management |
| `frontend` | 3000 | Next.js web application |

Infrastructure services: PostgreSQL (5432), Redis (6379), RabbitMQ (5672/15672), ClamAV (3310), Kong (8000), Supabase Auth.

### Backend Code Organization

```
backend/backend/
├── blocks/              # ~60+ reusable workflow blocks (the core integration library)
├── server/              # REST/WebSocket API implementation
│   ├── routers/         # API route handlers (v1.py is main router, 36K+ lines)
│   ├── middleware/       # Security middleware (cache control)
│   └── integrations/    # OAuth route handlers
├── executor/            # Graph execution engine
├── data/                # Data access layer, Pydantic models
├── integrations/        # OAuth providers (GitHub, Google, Twitter, etc.)
├── notifications/       # Notification delivery system
├── usecases/            # Business logic layer
├── monitoring/          # Prometheus metrics
└── util/                # Settings, helpers
```

### Frontend Code Organization

```
frontend/src/
├── app/                 # Next.js App Router
│   ├── (platform)/      # Main platform pages (builder, marketplace, monitor, etc.)
│   ├── (no-navbar)/     # Login/signup pages (no navbar layout)
│   └── api/             # API routes + auto-generated client
├── components/          # React components organized by domain
│   ├── ui/              # Radix UI base primitives
│   ├── agptui/          # AutoGPT-specific UI components
│   ├── edit/            # Workflow builder UI
│   └── ...              # admin/, agents/, auth/, marketplace/, etc.
├── hooks/               # Custom React hooks (useAgentGraph is central)
├── lib/                 # Utilities
│   ├── autogpt-server-api/  # API client, context, types
│   ├── supabase/        # Auth middleware
│   └── react-query/     # TanStack Query config
├── services/            # Business logic services
└── tests/               # Playwright E2E test suites
```

### Key Concepts

1. **Agent Graphs**: Workflow definitions composed of connected blocks, stored as JSON, executed by the backend executor
2. **Blocks**: Reusable Python components in `/backend/backend/blocks/` -- each performs a specific task (LLM calls, API integrations, data transformation). ~60+ block types across 27+ integration categories
3. **Block Schema**: Blocks define typed Input/Output schemas using Pydantic, enabling the visual graph editor to render correct UIs
4. **Integrations**: OAuth and API credential management, stored per user
5. **Store/Marketplace**: Users can publish and share agent templates
6. **Execution**: Graphs execute node-by-node with streaming output via WebSocket

### Database

- **Schema file:** `/autogpt_platform/backend/schema.prisma` (846 lines, 40+ models)
- **Key models:** User, AgentGraph (versioned), AgentNode, AgentNodeLink, AgentGraphExecution, AgentNodeExecution, StoreListing, APIKey, CreditTransaction
- **Migrations:** 75+ Prisma migration files in `/backend/migrations/`
- **Features:** pgvector for embeddings, materialized views for analytics

## Testing Approach

### Backend Testing
- Framework: pytest with pytest-asyncio (auto async mode)
- Test files are **colocated** with source: `module.py` -> `module_test.py`
- Snapshot testing via pytest-snapshot for API response validation
- Block tests parametrically validate all block types: `test_available_blocks`
- Run `poetry run test` which uses a Docker-based isolated test database
- To update snapshots: `poetry run pytest path/to/test.py --snapshot-update`
- Always review snapshot diffs with `git diff` before committing

### Frontend Testing
- Framework: Playwright for E2E tests
- Tests in `/frontend/src/tests/` using `*.spec.ts` pattern
- Page Object Model pattern for test organization
- Storybook for component isolation and visual testing
- MSW (Mock Service Worker) for API mocking in stories
- Run `pnpm test` (includes build) or `pnpm test:no-build`

## Environment Configuration

### Configuration Files
- **Platform root:** `/autogpt_platform/.env.default` -> `.env` (Supabase/shared)
- **Backend:** `/autogpt_platform/backend/.env.default` -> `.env`
- **Frontend:** `/autogpt_platform/frontend/.env.default` -> `.env`

### Loading Priority (highest to lowest)
1. Shell environment variables
2. Docker Compose `environment:` sections
3. `.env` files (user overrides, gitignored)
4. `.env.default` files (defaults, tracked in git)

### Key Environment Variables
- `DATABASE_URL`, `DIRECT_URL` -- PostgreSQL connection strings
- `REDIS_HOST`, `REDIS_PORT` -- Redis connection
- `RABBITMQ_DEFAULT_USER`, `RABBITMQ_DEFAULT_PASS` -- RabbitMQ credentials
- `SUPABASE_JWT_SECRET` -- JWT authentication secret
- `ENCRYPTION_KEY` -- Fernet key for credential encryption
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc. -- LLM provider keys
- `NEXT_PUBLIC_AGPT_SERVER_URL` -- Frontend -> Backend API URL
- `NEXT_PUBLIC_AGPT_WS_SERVER_URL` -- Frontend -> WebSocket URL

## Common Development Tasks

### Adding a New Block

1. Create a new file in `/autogpt_platform/backend/backend/blocks/`
2. Inherit from `Block` base class
3. Define `Input` and `Output` schemas using `BlockSchema`
4. Implement async `run` method yielding `(pin_name, value)` tuples
5. Generate a UUID with `uuid.uuid4()` for the block ID
6. The block auto-registers via dynamic discovery
7. Run block tests: `poetry run pytest backend/blocks/test/test_block.py -xvs`
8. Consider whether input/output types connect well with other blocks in the graph editor

### Modifying the API

1. Update routes in `/backend/backend/server/routers/`
2. Add/update Pydantic models
3. Write colocated tests (`*_test.py` beside the route file)
4. Run `poetry run test`
5. Frontend API client will auto-regenerate on next `pnpm dev` or `pnpm generate:api`

### Frontend Feature Development

1. Components go in `/frontend/src/components/` organized by domain
2. Use existing Radix UI primitives from `/frontend/src/components/ui/`
3. Add Storybook stories for new components
4. For protected routes, update `/frontend/src/lib/supabase/middleware.ts`
5. Test with Playwright for user-facing flows

### Database Schema Changes

1. Edit `/autogpt_platform/backend/schema.prisma`
2. Run `poetry run prisma migrate dev --name describe_change`
3. Run `poetry run prisma generate` to update the client
4. Update data access layer in `/backend/backend/data/`

## Pre-commit Hooks

The repo uses pre-commit hooks (`.pre-commit-config.yaml`) that enforce:

- **General:** Large file detection, byte order mark fixing, merge conflict detection, secret detection
- **Python (Backend):** Ruff linting, Black formatting, isort import ordering, Pyright type checking
- **Python (Classic):** Flake8, Black, isort, Pyright
- **Frontend:** Prettier formatting, TypeScript type checking
- **Infrastructure:** Prisma client generation on schema changes, Poetry dependency validation

## Code Style and Conventions

### Python (Backend)
- Formatter: Black (default settings)
- Import sorting: isort with Black profile
- Linting: Ruff (target Python 3.10)
- Type checking: Pyright
- Async: Use `async def` for FastAPI endpoints and block `run` methods
- Tests: pytest with `asyncio_mode = "auto"` (no need for `@pytest.mark.asyncio`)
- Dependencies: Alphabetical order in `pyproject.toml` (noted in file comments)

### TypeScript (Frontend)
- Strict TypeScript mode enabled
- ESLint extends: `next/core-web-vitals`, `next/typescript`, `@tanstack/query/recommended`
- `react-hooks/exhaustive-deps` is OFF (rely on code review)
- `@typescript-eslint/no-explicit-any` is OFF (handled case-by-case)
- Unused vars with `_` prefix are allowed
- Prettier with `prettier-plugin-tailwindcss` for auto-sorting Tailwind classes
- Path alias: `@/*` maps to `./src/*`

### Commit Messages

Use conventional commit format:

**Types:** `feat`, `fix`, `refactor`, `ci`, `docs`, `dx`

**Scopes:**
- `platform` -- changes affecting both frontend and backend
- `frontend`, `backend` -- single-side changes
- `blocks` -- block additions/modifications
- `infra` -- infrastructure changes
- Sub-scopes: `backend/executor`, `backend/db`, `frontend/builder`, `infra/prod`

**Examples:**
```
feat(blocks): add Airtable integration block
fix(backend/executor): handle timeout in graph execution
refactor(frontend/builder): simplify node connection logic
dx(platform): update Docker Compose for faster local dev
```

## Pull Request Guidelines

- Target the `dev` branch (not `master`)
- Use conventional commit format for PR titles
- Fill out `.github/PULL_REQUEST_TEMPLATE.md` (Changes section + checklist)
- Keep out-of-scope changes under 20% of the PR
- For `data/*.py` changes, validate user ID checks or explain omission
- For new protected frontend routes, update Supabase middleware
- For configuration changes, update `.env.default` and `docker-compose.yml`
- Run pre-commit hooks before submitting

## Reviewing Pull Requests

```bash
# Get PR reviews
gh api /repos/Significant-Gravitas/AutoGPT/pulls/{pr_number}/reviews

# Get review comments
gh api /repos/Significant-Gravitas/AutoGPT/pulls/{pr_number}/reviews/{review_id}/comments

# Get PR comments
gh api /repos/Significant-Gravitas/AutoGPT/issues/{pr_number}/comments

# Get PR diff
gh pr diff {pr_number}
```

## License Notes

- Code in `autogpt_platform/` is under **Polyform Shield License** -- contributors must sign a CLA
- Everything else (classic/, docs/, etc.) is under **MIT License**
- Do not mix code across license boundaries without consideration
