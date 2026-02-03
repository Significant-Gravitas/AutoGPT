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
│   ├── backend/                # Python FastAPI backend (v0.4.9)
│   ├── frontend/               # Next.js React frontend (v0.3.4)
│   ├── autogpt_libs/           # Shared Python libraries
│   ├── db/                     # Supabase/PostgreSQL Docker config
│   ├── graph_templates/        # Predefined agent workflow templates
│   ├── installer/              # Installation scripts (Linux + Windows)
│   ├── docker-compose.yml      # Service orchestration
│   └── docker-compose.platform.yml
├── classic/                    # Classic AutoGPT (MIT License)
│   ├── original_autogpt/       # Original CLI-based agent
│   ├── forge/                  # Agent development framework
│   ├── benchmark/              # Agent benchmarking (agbenchmark)
│   └── frontend/               # Flutter desktop UI
├── docs/                       # MkDocs documentation site
├── .github/                    # CI/CD workflows, PR templates, CODEOWNERS
└── .pre-commit-config.yaml     # Pre-commit hooks
```

## Tech Stack Summary

### Platform Backend (`autogpt_platform/backend/`)
- **Language:** Python 3.10-3.12
- **Framework:** FastAPI 0.116+ with Uvicorn
- **ORM:** Prisma 0.15+ (async Python client)
- **Database:** PostgreSQL with pgvector
- **Queue:** RabbitMQ (aio-pika)
- **Cache:** Redis
- **Auth:** Supabase + JWT
- **Package Manager:** Poetry 2.1
- **Testing:** pytest, pytest-asyncio, pytest-snapshot
- **Linting/Formatting:** Black, isort, Ruff, Pyright
- **Monitoring:** Sentry SDK (with FastAPI, OpenAI, Anthropic, LaunchDarkly integrations)

### Platform Frontend (`autogpt_platform/frontend/`)
- **Framework:** Next.js 15.4 (App Router)
- **Language:** TypeScript 5.9
- **UI Library:** React 18.3
- **Styling:** Tailwind CSS 3.4
- **Components:** Radix UI primitives (shadcn/ui pattern)
- **State Management:** React hooks, TanStack Query 5.85, Supabase client
- **Workflow Editor:** @xyflow/react 12.8
- **Package Manager:** pnpm 10.11+
- **Testing:** Playwright 1.54 (E2E), Storybook 9.1 (component)
- **Linting/Formatting:** ESLint, Prettier (with Tailwind plugin)
- **API Client:** Auto-generated via Orval 7.11 from OpenAPI spec

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
poetry run rest          # REST API (port 8006)
poetry run ws            # WebSocket server (port 8001)
poetry run executor      # Execution engine
poetry run scheduler     # Task scheduler
poetry run notification  # Notification service
poetry run db            # Database utilities
poetry run cli           # CLI tools

# Run all tests
poetry run test

# Run specific test file
poetry run pytest path/to/test_file.py::test_function_name

# Run block tests (validates all block types)
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

# Generate Playwright test code
pnpm gentests

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

# Force regenerate API client
pnpm generate:api:force
```

### Docker Services

```bash
cd autogpt_platform

# Start all services
docker compose up -d

# Start with extended platform services (includes Supabase Studio)
docker compose -f docker-compose.yml -f docker-compose.platform.yml up -d

# Scale executor service
docker compose up -d --scale executor=3

# View logs
docker compose logs -f rest_server executor

# Rebuild and restart specific service
docker compose build rest_server && docker compose up -d --no-deps rest_server

# Watch mode (auto-update on code changes)
docker compose watch
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

The platform runs as multiple microservices communicating via RabbitMQ and Redis:

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
├── blocks/              # Reusable workflow blocks (25+ integration packages)
│   ├── block.py         # Block base class and registry
│   ├── basic.py         # Core utility blocks
│   ├── llm.py           # LLM integration blocks (OpenAI, Anthropic, Groq, Ollama)
│   ├── http.py          # HTTP request blocks
│   ├── github/          # GitHub integration (issues, PRs, CI, reviews, triggers)
│   ├── google/          # Google services (Calendar, Gmail, Sheets)
│   ├── twitter/         # Twitter/X integration
│   ├── airtable/        # Airtable integration (records, schema, triggers)
│   ├── hubspot/         # HubSpot CRM (contacts, companies, engagements)
│   ├── exa/             # Exa search (search, answers, websets, webhooks)
│   ├── firecrawl/       # Web scraping (crawl, scrape, extract, map, search)
│   ├── apollo/          # Apollo.io (people, organizations)
│   ├── ayrshare/        # Social media posting (13 platforms)
│   ├── fal/             # AI video generation
│   ├── discord.py       # Discord integration
│   ├── reddit.py        # Reddit integration
│   ├── medium.py        # Medium publishing
│   ├── youtube.py       # YouTube transcripts
│   └── ...              # 40+ more block files + test/
├── server/
│   ├── rest_api.py      # FastAPI app setup, CORS, middleware
│   ├── ws_api.py        # WebSocket server setup
│   ├── routers/
│   │   ├── v1.py        # Main API router (~1150 lines)
│   │   ├── analytics.py # Analytics endpoints
│   │   └── postmark/    # Postmark email webhook handlers
│   ├── middleware/
│   │   └── security.py  # Cache-control security middleware
│   ├── integrations/    # OAuth callback route handlers
│   └── v2/             # V2 API layer
│       ├── store/       # Enhanced store with media upload
│       ├── library/     # Library routes (agents, presets)
│       ├── admin/       # Credit and store management
│       ├── otto/        # Otto AI integration
│       ├── AutoMod/     # Automation/moderation
│       └── turnstile/   # Cloudflare Turnstile CAPTCHA
├── executor/
│   ├── manager.py       # ExecutionManager: orchestrates graph execution
│   ├── scheduler.py     # Scheduler: handles cron-triggered runs
│   └── database.py      # DatabaseManager: connection pooling
├── data/                # Data access layer
│   ├── graph.py         # AgentGraph CRUD operations
│   ├── execution.py     # Execution record management
│   ├── block.py         # Block base class, schemas, registry
│   ├── cost.py          # Cost tracking system
│   ├── credit.py        # Credit/billing management
│   ├── api_key.py       # API key management
│   └── ...              # analytics, notifications, onboarding, integrations
├── integrations/
│   └── oauth/           # OAuth providers (GitHub, Google, Twitter, Notion, Todoist)
├── notifications/       # Notification delivery system
│   ├── email.py         # Email via Postmark
│   └── templates/       # Jinja2 templates (agent_run, low_balance, refund, weekly_summary)
├── usecases/            # Business logic layer
│   ├── block_autogen.py # Block auto-generation
│   ├── reddit_marketing.py
│   └── router.py        # Use case routing
├── monitoring/          # Prometheus metrics
│   ├── block_error_monitor.py
│   ├── late_execution_monitor.py
│   └── notification_monitor.py
└── util/
    ├── settings.py      # Pydantic-based settings management
    ├── cloud_storage.py # Google Cloud Storage integration
    ├── virus_scanner.py # ClamAV virus scanning
    ├── feature_flag.py  # LaunchDarkly feature flags
    ├── encryption.py    # Fernet encryption
    └── ...              # logging, metrics, retry, request helpers
```

### Frontend Code Organization

```
frontend/src/
├── app/                        # Next.js 15 App Router
│   ├── layout.tsx              # Root layout
│   ├── providers.tsx           # Provider hierarchy
│   ├── globals.css             # Global Tailwind styles
│   ├── (platform)/             # Main platform pages (require auth)
│   │   ├── admin/              # Admin dashboard
│   │   ├── auth/               # Auth pages
│   │   ├── build/              # Visual workflow builder
│   │   ├── library/            # User's agent library
│   │   ├── marketplace/        # Agent marketplace/store
│   │   ├── monitoring/         # Execution monitoring
│   │   ├── profile/            # User profile
│   │   ├── dictionaries/       # Dictionary management
│   │   └── health/             # Health check
│   ├── (no-navbar)/            # Auth pages (login, signup, reset-password)
│   └── api/                    # Next.js API routes + auto-generated client
├── components/                 # React components organized by domain
│   ├── ui/                     # Radix UI base primitives (shadcn/ui)
│   ├── atoms/                  # Small reusable components
│   ├── molecules/              # Compound components
│   ├── agptui/                 # AutoGPT-specific UI components
│   ├── edit/                   # Workflow builder UI
│   ├── admin/                  # Admin panel components
│   ├── agents/                 # Agent-related components
│   ├── auth/                   # Auth UI components
│   ├── marketplace/            # Marketplace components
│   ├── integrations/           # Credential management UI
│   ├── runner-ui/              # Agent execution interface
│   ├── layout/                 # Navbar, sidebar layout
│   ├── onboarding/             # Onboarding flow
│   ├── analytics/              # Analytics components
│   ├── contextual/             # Contextual components
│   ├── tokens/                 # Token-related components
│   └── monitor/                # Monitor components
├── hooks/
│   ├── useAgentGraph.tsx       # Central workflow state management hook
│   ├── useCredentials.ts       # Credential management
│   ├── useCredits.ts           # Credit balance tracking
│   ├── useCopyPaste.ts         # Copy/paste functionality
│   ├── useBezierPath.ts        # Edge path calculations
│   └── useTurnstile.ts         # CAPTCHA integration
├── lib/
│   ├── autogpt-server-api/     # Backend API client, context, types
│   ├── supabase/               # Auth client and middleware
│   ├── react-query/            # TanStack Query configuration
│   ├── utils/                  # General utilities
│   └── turnstile.ts            # Turnstile utilities
├── services/                   # Feature flag service, storage service
└── tests/                      # Playwright E2E test suites (20+ spec files)
```

### Key Concepts

1. **Agent Graphs**: Workflow definitions composed of connected blocks, stored as JSON, executed by the backend executor
2. **Blocks**: Reusable Python components in `/backend/backend/blocks/` -- each performs a specific task (LLM calls, API integrations, data transformation). 25+ integration packages with many block types each
3. **Block Schema**: Blocks define typed Input/Output schemas using Pydantic, enabling the visual graph editor to render correct UIs
4. **Integrations**: OAuth and API credential management, stored per user (GitHub, Google, Twitter, Notion, Todoist, and more)
5. **Store/Marketplace**: Users can publish and share agent templates (v2 API with enhanced search/filtering)
6. **Execution**: Graphs execute node-by-node with streaming output via WebSocket
7. **Credits System**: Stripe-powered payment system with block cost configuration
8. **Feature Flags**: LaunchDarkly integration for gradual feature rollouts
9. **Otto**: AI assistant integration within the platform

### Database

- **Schema file:** `/autogpt_platform/backend/schema.prisma` (845 lines, 24 models)
- **Key models:** User, UserOnboarding, AgentGraph (versioned), AgentNode, AgentNodeLink, AgentBlock, AgentGraphExecution, AgentNodeExecution, AgentPreset, LibraryAgent, StoreListing, StoreListingVersion, StoreListingReview, APIKey, CreditTransaction, CreditRefundRequest, Profile, IntegrationWebhook, NotificationEvent, AnalyticsDetails, AnalyticsMetrics
- **Migrations:** 74 Prisma migration files in `/backend/migrations/`
- **Features:** pgvector for embeddings, materialized views for analytics, composite indexes

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
- Framework: Playwright 1.54 for E2E tests
- Tests in `/frontend/src/tests/` using `*.spec.ts` pattern (20+ test suites)
- Page Object Model pattern for test organization
- Global setup creates test users dynamically
- Storybook 9.1 for component isolation and visual testing
- MSW (Mock Service Worker) for API mocking in stories
- Chromatic for visual regression testing
- axe-playwright for accessibility testing
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
- `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` -- Redis connection
- `RABBITMQ_DEFAULT_USER`, `RABBITMQ_DEFAULT_PASS` -- RabbitMQ credentials
- `SUPABASE_JWT_SECRET` -- JWT authentication secret
- `ENCRYPTION_KEY` -- Fernet key for credential encryption
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY` -- LLM provider keys
- `STRIPE_API_KEY`, `STRIPE_WEBHOOK_SECRET` -- Payment processing
- `POSTMARK_SERVER_API_TOKEN` -- Email notifications
- `NEXT_PUBLIC_AGPT_SERVER_URL` -- Frontend -> Backend API URL
- `NEXT_PUBLIC_AGPT_WS_SERVER_URL` -- Frontend -> WebSocket URL
- `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` -- Supabase config

## Common Development Tasks

### Adding a New Block

1. Create a new file in `/autogpt_platform/backend/backend/blocks/` (or a subdirectory for multi-file integrations)
2. Inherit from `Block` base class
3. Define `Input` and `Output` schemas using `BlockSchema` with `SchemaField` descriptors
4. Implement async `run` method yielding `(pin_name, value)` tuples
5. Generate a UUID with `uuid.uuid4()` for the block ID
6. The block auto-registers via dynamic discovery (`load_all_blocks()`)
7. Run block tests: `poetry run pytest backend/blocks/test/test_block.py -xvs`
8. Consider whether input/output types connect well with other blocks in the graph editor

### Modifying the API

1. Update routes in `/backend/backend/server/routers/` (v1.py is the main router, v2/ for new features)
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

- **General:** Large file detection (500KB max), byte order mark fixing, merge conflict detection, secret detection
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
- Sub-scopes: `backend/executor`, `backend/db`, `frontend/builder`, `frontend/library`, `frontend/marketplace`, `infra/prod`

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
