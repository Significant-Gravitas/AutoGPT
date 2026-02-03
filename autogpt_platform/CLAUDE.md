# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the AutoGPT Platform.

For repository-wide context, see the root `/CLAUDE.md`.

## Platform Overview

AutoGPT Platform is a graph-based workflow automation system for building, deploying, and managing AI agents. It consists of:

- **Backend** (`/backend`): Python FastAPI server (v0.4.9) with async support
- **Frontend** (`/frontend`): Next.js React application (v0.3.4)
- **Shared Libraries** (`/autogpt_libs`): Common Python utilities
- **Graph Templates** (`/graph_templates`): Pre-built agent workflow templates (Discord bots, Medium blogger)
- **Installer** (`/installer`): Automated setup scripts for Linux and Windows

## Essential Commands

### Backend Development

```bash
cd backend

# Install dependencies
poetry install

# Generate Prisma client (required after schema changes)
poetry run prisma generate

# Run database migrations
poetry run prisma migrate dev

# Start all infrastructure services (PostgreSQL, Redis, RabbitMQ, ClamAV)
docker compose -f ../docker-compose.yml up -d

# Run the backend (all services together)
poetry run app

# Run individual services
poetry run rest          # REST API server (port 8006)
poetry run ws            # WebSocket server (port 8001)
poetry run executor      # Graph execution engine
poetry run scheduler     # Cron-triggered task scheduler
poetry run notification  # Notification service
poetry run db            # Database utilities
poetry run cli           # CLI tools

# Run all tests (uses isolated Docker-based test database)
poetry run test

# Run specific test file or function
poetry run pytest path/to/test_file.py::test_function_name

# Run block tests (validates all block types)
poetry run pytest backend/blocks/test/test_block.py -xvs

# Run test for a specific block (e.g., GetCurrentTimeBlock)
poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[GetCurrentTimeBlock]' -xvs

# Format code (Black + isort)
poetry run format

# Lint code (Ruff)
poetry run lint
```

#### Snapshot Testing

When you first write a test or when expected output changes:

```bash
poetry run pytest path/to/test.py --snapshot-update
```

**Important**: Always review snapshot changes with `git diff` before committing. Snapshots are stored in `snapshots/` directories next to test files.

### Frontend Development

The frontend uses **pnpm** (not npm). Ensure pnpm 10.11+ is available via corepack.

```bash
cd frontend

# Install dependencies
pnpm install

# Start dev server (Turbo mode, auto-generates API client)
pnpm dev

# Build for production
pnpm build

# Run E2E tests (builds first, then runs Playwright)
pnpm test

# Run E2E tests without rebuilding
pnpm test:no-build

# Interactive Playwright test UI
pnpm test-ui

# Generate test code with Playwright codegen
pnpm gentests

# Lint (ESLint + Prettier check)
pnpm lint

# Format (ESLint fix + Prettier write)
pnpm format

# Type check only
pnpm types

# Storybook for component development (port 6006)
pnpm storybook

# Build Storybook
pnpm build-storybook

# Regenerate API client from backend OpenAPI spec (backend must be running)
pnpm generate:api

# Force regenerate API client
pnpm generate:api:force
```

### Docker Compose

```bash
# Start all services
docker compose up -d

# Start with extended platform services (includes Supabase Studio)
docker compose -f docker-compose.yml -f docker-compose.platform.yml up -d

# Scale executor for load testing
docker compose up -d --scale executor=3

# View service logs
docker compose logs -f rest_server executor

# Rebuild and restart a specific service
docker compose build rest_server && docker compose up -d --no-deps rest_server

# Full restart (stop, remove, pull, restart)
docker compose stop && docker compose rm -f && docker compose pull && docker compose up -d

# Watch mode (auto-update on code changes)
docker compose watch
```

## Architecture

### Backend Services

The backend runs as multiple independent microservices communicating via RabbitMQ and Redis:

| Service | Entry Point | Port | Purpose |
|---------|-------------|------|---------|
| `rest_server` | `backend.rest:main` | 8006 | FastAPI REST API |
| `websocket_server` | `backend.ws:main` | 8001 | Real-time WebSocket updates |
| `executor` | `backend.exec:main` | 8002 | Agent graph execution engine |
| `scheduler_server` | `backend.scheduler:main` | 8003 | Cron-triggered execution |
| `notification_server` | `backend.notification:main` | 8007 | Email/webhook notifications |
| `database_manager` | `backend.db:main` | 8005 | Database connection management |

Infrastructure: PostgreSQL (5432), Redis (6379), RabbitMQ (5672/15672), ClamAV (3310), Kong API Gateway (8000), Supabase Auth.

### Backend Code Structure

```
backend/backend/
├── app.py                # Main entry: runs all services together
├── rest.py               # REST-only entry point
├── ws.py                 # WebSocket-only entry point
├── exec.py               # Executor entry point
├── scheduler.py          # Scheduler entry point
├── notification.py       # Notification entry point
├── blocks/               # Reusable workflow blocks (25+ integration packages)
│   ├── block.py          # Block base class and registry
│   ├── basic.py          # Core utility blocks
│   ├── llm.py            # LLM integration blocks
│   ├── http.py           # HTTP request blocks
│   ├── github/           # GitHub (issues, PRs, CI, reviews, triggers)
│   ├── google/           # Google (Calendar, Gmail, Sheets)
│   ├── twitter/          # Twitter/X integration
│   ├── airtable/         # Airtable (records, schema, bases, triggers)
│   ├── hubspot/          # HubSpot CRM (contacts, companies, engagements)
│   ├── exa/              # Exa search (search, answers, websets, webhooks)
│   ├── firecrawl/        # Web scraping (crawl, scrape, extract, map, search)
│   ├── apollo/           # Apollo.io (people, organizations)
│   ├── ayrshare/         # Social media posting (13 platforms)
│   ├── fal/              # AI video generation
│   ├── linear/           # Linear project management
│   ├── jina/             # Jina AI (chunking)
│   ├── nvidia/           # NVIDIA integration
│   ├── slant3d/          # 3D printing integration
│   ├── smartlead/        # SmartLead email outreach
│   ├── enrichlayer/      # Data enrichment (LinkedIn)
│   ├── wordpress/        # WordPress integration
│   ├── wolfram/          # Wolfram Alpha
│   ├── todoist/          # Todoist task management
│   ├── replicate/        # Replicate AI models
│   ├── compass/          # Compass triggers
│   ├── generic_webhook/  # Generic webhook triggers
│   ├── discord.py        # Discord
│   ├── reddit.py         # Reddit
│   ├── medium.py         # Medium publishing
│   ├── youtube.py        # YouTube transcripts
│   ├── pinecone.py       # Pinecone vector DB
│   ├── mem0.py           # Mem0 memory
│   ├── ideogram.py       # Ideogram image generation
│   ├── flux_kontext.py   # Flux Kontext image generation
│   ├── ai_image_generator_block.py
│   ├── ai_music_generator.py
│   ├── ai_shortform_video_block.py
│   ├── text_to_speech_block.py
│   ├── talking_head.py   # Talking head video
│   ├── screenshotone.py  # Screenshot service
│   ├── ...               # More utility blocks
│   └── test/             # Block test suite
├── server/
│   ├── rest_api.py       # FastAPI app setup, CORS, middleware
│   ├── ws_api.py         # WebSocket server setup
│   ├── routers/
│   │   ├── v1.py         # Main API router (~1150 lines)
│   │   ├── analytics.py  # Analytics endpoints
│   │   └── postmark/     # Postmark email webhook handlers
│   ├── middleware/
│   │   └── security.py   # Cache-control security middleware
│   ├── integrations/     # OAuth callback route handlers
│   └── v2/              # V2 API layer
│       ├── store/        # Enhanced store with media upload to GCS
│       ├── library/      # Library routes (agents, presets)
│       ├── admin/        # Credit and store management
│       ├── otto/         # Otto AI integration
│       ├── AutoMod/      # Automation/moderation
│       └── turnstile/    # Cloudflare Turnstile CAPTCHA
├── executor/
│   ├── manager.py        # ExecutionManager: orchestrates graph execution
│   ├── scheduler.py      # Scheduler: handles cron-triggered runs
│   ├── database.py       # DatabaseManager: connection pooling
│   └── utils.py          # Execution utilities
├── data/                 # Data access layer
│   ├── graph.py          # AgentGraph CRUD operations
│   ├── execution.py      # Execution record management
│   ├── block.py          # Block base class, schemas, registry
│   ├── cost.py           # Cost tracking system
│   ├── block_cost_config.py # Block cost configuration
│   ├── credit.py         # Credit/billing management
│   ├── api_key.py        # API key management
│   ├── analytics.py      # Analytics data access
│   ├── integrations.py   # Integration data access
│   ├── notifications.py  # Notification data access
│   ├── onboarding.py     # Onboarding data access
│   ├── user.py           # User data access
│   ├── model.py          # Shared Pydantic models
│   ├── db.py             # Database utilities
│   ├── rabbitmq.py       # RabbitMQ client
│   ├── redis_client.py   # Redis client
│   └── event_bus.py      # Event bus
├── integrations/
│   └── oauth/            # OAuth providers (GitHub, Google, Twitter, Notion, Todoist)
├── notifications/        # Notification delivery system
│   ├── email.py          # Email via Postmark
│   ├── notifications.py  # Notification logic
│   └── templates/        # Jinja2 email templates
│       ├── base.html.jinja2
│       ├── agent_run.html.jinja2
│       ├── low_balance.html.jinja2
│       ├── refund_processed.html.jinja2
│       ├── refund_request.html.jinja2
│       └── weekly_summary.html.jinja2
├── usecases/             # Business logic layer
│   ├── block_autogen.py  # Block auto-generation
│   ├── reddit_marketing.py
│   ├── sample.py
│   ├── models.py         # Use case models
│   └── router.py         # Use case routing
├── monitoring/           # Prometheus metrics collection
│   ├── block_error_monitor.py
│   ├── late_execution_monitor.py
│   └── notification_monitor.py
└── util/
    ├── settings.py       # Pydantic-based settings management
    ├── cloud_storage.py  # Google Cloud Storage integration
    ├── virus_scanner.py  # ClamAV virus scanning (async)
    ├── feature_flag.py   # LaunchDarkly feature flags
    ├── encryption.py     # Fernet encryption
    ├── logging.py        # Logging configuration
    ├── metrics.py        # Prometheus metrics
    ├── retry.py          # Retry logic
    ├── request.py        # HTTP request helpers
    ├── file.py           # File utilities
    ├── text.py           # Text utilities
    ├── truncate.py       # Truncation utilities
    ├── decorator.py      # Decorators
    ├── data.py           # Data utilities
    ├── json.py           # JSON utilities
    ├── process.py        # Process management
    ├── prompt.py         # Prompt utilities
    ├── service.py        # Service base class
    ├── clients.py        # API clients
    ├── models.py         # Shared models
    ├── type.py           # Type utilities
    ├── mock.py           # Test mocking utilities
    └── exceptions.py     # Custom exceptions
```

### Frontend Architecture

```
frontend/src/
├── app/                        # Next.js 15 App Router
│   ├── layout.tsx              # Root layout
│   ├── providers.tsx           # Provider hierarchy (React Query, Theme, Auth, etc.)
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
│   │   ├── health/             # Health check
│   │   ├── login/              # Login page
│   │   ├── signup/             # Signup page
│   │   ├── reset-password/     # Password reset
│   │   └── unauthorized/       # Unauthorized access page
│   ├── (no-navbar)/            # Auth pages (no navbar layout)
│   └── api/                    # Next.js API routes + auto-generated client
│       └── __generated__/      # Orval-generated React Query hooks
├── components/                 # React components by domain
│   ├── ui/                     # Radix UI base primitives (shadcn/ui pattern)
│   ├── atoms/                  # Small reusable components (Badge, Button, Input, etc.)
│   ├── molecules/              # Compound components
│   ├── agptui/                 # AutoGPT-specific compound components
│   ├── edit/                   # Workflow builder components
│   ├── admin/                  # Admin panel components
│   ├── agents/                 # Agent-related components
│   ├── auth/                   # Auth UI components
│   ├── marketplace/            # Marketplace components
│   ├── integrations/           # Credential management UI
│   ├── runner-ui/              # Agent execution interface
│   ├── layout/                 # Navbar, sidebar layout components
│   ├── onboarding/             # Onboarding flow
│   ├── analytics/              # Analytics visualizations
│   ├── contextual/             # Contextual components
│   ├── tokens/                 # Token-related components
│   ├── monitor/                # Monitoring components
│   ├── styles/                 # Style components
│   ├── OttoChatWidget.tsx      # Otto AI chat widget
│   ├── CustomNode.tsx          # Graph node component
│   ├── CustomEdge.tsx          # Graph edge component
│   ├── Flow.tsx                # Main flow editor
│   ├── NodeHandle.tsx          # Node connection handles
│   ├── DataTable.tsx           # Generic data table
│   └── RoleBasedAccess.tsx     # Role-based access control
├── hooks/
│   ├── useAgentGraph.tsx       # Central workflow state management hook
│   ├── useCredentials.ts       # Credential management
│   ├── useCredits.ts           # Credit balance tracking
│   ├── useCopyPaste.ts         # Copy/paste functionality
│   ├── useBezierPath.ts        # Edge path calculations
│   └── useTurnstile.ts         # Cloudflare Turnstile CAPTCHA
├── lib/
│   ├── autogpt-server-api/     # Backend API client, context provider, types
│   ├── supabase/               # Auth client and middleware
│   ├── react-query/            # TanStack Query configuration
│   ├── utils/                  # General utilities
│   ├── utils.ts                # Core utility functions
│   ├── turnstile.ts            # Turnstile utilities
│   ├── env-config.ts           # Environment configuration
│   ├── cron-expression-utils.ts # Cron expression parsing
│   └── withRoleAccess.ts       # Role-based access HOC
├── services/
│   ├── feature-flags/          # LaunchDarkly feature flag service
│   └── storage/                # Storage service
├── tests/                      # Playwright E2E tests (20+ spec files)
│   ├── *.spec.ts               # Test suites
│   ├── pages/                  # Page Object Model classes
│   ├── utils/                  # Test utilities
│   └── global-setup.ts         # Global test setup (creates test users)
└── types/                      # Shared TypeScript types
```

**Provider Hierarchy** (in `providers.tsx`):
QueryClientProvider -> NuqsAdapter -> NextThemesProvider -> BackendAPIProvider -> CredentialsProvider -> LaunchDarklyProvider -> OnboardingProvider -> TooltipProvider

**Key Frontend Technologies:**
- React 18.3 with Next.js 15.4 App Router
- TypeScript 5.9 (strict mode)
- Tailwind CSS 3.4 for styling
- Radix UI for accessible primitives
- @xyflow/react 12.8 for the visual workflow builder
- TanStack Query 5.85 for server state
- React Hook Form + Zod for forms
- Supabase for auth and real-time
- LaunchDarkly for feature flags
- Orval 7.11 for auto-generated API client from OpenAPI spec
- Framer Motion for animations
- Recharts for data visualization
- Sentry for error monitoring

### Key Concepts

1. **Agent Graphs**: Workflow definitions stored as JSON, executed by the backend
2. **Blocks**: Reusable components in `/backend/blocks/` that perform specific tasks (25+ integration packages)
3. **Integrations**: OAuth and API connections stored per user (GitHub, Google, Twitter, Notion, Todoist, and 20+ more providers)
4. **Store**: Marketplace for sharing agent templates (v2 API with enhanced search/filtering)
5. **Virus Scanning**: ClamAV integration for file upload security
6. **Credits System**: Stripe-powered payment system with block cost configuration
7. **Notifications**: Email notification system via Postmark with Jinja2 templates for agent runs, low balance, refunds, and weekly summaries
8. **Feature Flags**: LaunchDarkly integration for gradual feature rollouts
9. **Otto**: AI assistant integration within the platform (chat widget + backend service)

### Authentication

- JWT-based via Supabase integration
- FastAPI `Depends()` pattern with `auth_middleware` for route protection
- API keys with granular permissions (EXECUTE_GRAPH, READ_GRAPH, EXECUTE_BLOCK, READ_BLOCK)
- For data layer changes (`data/*.py`), always validate user ID checks or explain why not needed

### API v2 Architecture

The backend includes a v2 API layer at `/backend/backend/server/v2/`:

- **`/v2/store/`**: Enhanced store with paginated listings, media upload to Google Cloud Storage, image generation
- **`/v2/library/`**: Library routes for agents and presets management
- **`/v2/admin/`**: Admin routes for credit and store management
- **`/v2/otto/`**: Otto AI integration features (models, routes, service)
- **`/v2/AutoMod/`**: Automation/moderation features
- **`/v2/turnstile/`**: Cloudflare Turnstile CAPTCHA integration

### Security: Cache Protection Middleware

- Located in `/backend/backend/server/middleware/security.py`
- Default: Disables caching for ALL endpoints (`Cache-Control: no-store, no-cache, must-revalidate, private`)
- Uses an allow list -- only explicitly permitted paths can be cached
- Cacheable paths: static assets (`/static/*`, `/_next/static/*`), health checks, public store pages, documentation
- To allow caching for a new endpoint, add it to `CACHEABLE_PATHS` in the middleware
- Applied to both main API server and external API applications

### Monitoring System

Located in `/backend/backend/monitoring/`:
- `block_error_monitor.py`: Monitors block execution errors
- `late_execution_monitor.py`: Tracks delayed executions
- `notification_monitor.py`: Monitors notification delivery

### Cloud Storage

- Google Cloud Storage integration via `gcloud-aio-storage`
- Utility module at `/backend/backend/util/cloud_storage.py`
- Used for media uploads in store submissions

### Feature Flags

**Backend**: Feature flag utilities in `/backend/backend/util/feature_flag.py`

**Frontend**: LaunchDarkly integration:
```typescript
// Check flag value
const isEnabled = useGetFlag(Flag.AGENT_ACTIVITY);

// Protect component with HOC
export default withFeatureFlag(Flag.MY_FEATURE)(MyComponent);

// Mock mode for Playwright tests via NEXT_PUBLIC_PW_TEST
```

### Database Schema

**Schema file:** `/backend/schema.prisma` (845 lines, 24 models)

Key models:

| Model | Purpose |
|-------|---------|
| `User` | Authentication, profile, credits balance, notification preferences |
| `UserOnboarding` | Onboarding step tracking |
| `AgentGraph` | Workflow definitions with version control |
| `AgentNode` | Individual nodes within workflows |
| `AgentNodeLink` | Connections between nodes |
| `AgentBlock` | Block type definitions |
| `AgentGraphExecution` | Execution history and results |
| `AgentNodeExecution` | Individual node execution records |
| `AgentPreset` | Saved agent configurations |
| `LibraryAgent` | User's agent library entries |
| `StoreListing` | Marketplace agent listings |
| `StoreListingVersion` | Versioned store listing content |
| `StoreListingReview` | User reviews and ratings |
| `APIKey` | User API keys with granular permissions |
| `IntegrationWebhook` | Webhook configurations |
| `CreditTransaction` | Usage-based billing records |
| `CreditRefundRequest` | Refund request tracking |
| `Profile` | User profile details |
| `NotificationEvent` | Notification event records |
| `UserNotificationBatch` | Batched notification delivery |
| `AnalyticsDetails` / `AnalyticsMetrics` | Usage tracking and analytics |

Features: pgvector for embeddings, materialized views for analytics (mv_agent_run_counts, mv_review_stats), composite indexes.

**Migrations:** 74 migration files in `/backend/migrations/`.

### Environment Configuration

#### Configuration Files

- **Platform root:** `.env.default` (Supabase/shared defaults) -> `.env` (user overrides)
- **Backend:** `backend/.env.default` -> `backend/.env`
- **Frontend:** `frontend/.env.default` -> `frontend/.env`

#### Loading Priority (highest to lowest)

1. Shell environment variables
2. Docker Compose `environment:` sections
3. `.env` files (user overrides, gitignored)
4. `.env.default` files (defaults, tracked in git)

#### Key Environment Variables

**Database:** `DATABASE_URL`, `DIRECT_URL`, `DB_HOST`, `DB_USER`, `DB_PASS`, `DB_PORT`, `DB_NAME`
**Redis:** `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`
**RabbitMQ:** `RABBITMQ_DEFAULT_USER`, `RABBITMQ_DEFAULT_PASS`
**Auth:** `SUPABASE_JWT_SECRET`, `JWT_SECRET`
**Security:** `ENCRYPTION_KEY` (Fernet format)
**LLM Keys:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`
**URLs:** `PLATFORM_BASE_URL`, `FRONTEND_BASE_URL`
**Payment:** `STRIPE_API_KEY`, `STRIPE_WEBHOOK_SECRET`
**Email:** `POSTMARK_SERVER_API_TOKEN`, `POSTMARK_SENDER_EMAIL`
**Frontend:** `NEXT_PUBLIC_AGPT_SERVER_URL`, `NEXT_PUBLIC_AGPT_WS_SERVER_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`

Docker Compose uses hardcoded defaults (no `${VARIABLE}` substitutions). The `env_file` directive loads variables into containers at runtime.

## Common Development Tasks

### Adding a New Block

1. Create a new file in `/backend/backend/blocks/` (or a subdirectory for multi-file integrations)
2. Inherit from `Block` base class
3. Define `Input` and `Output` schemas using `BlockSchema` with `SchemaField` descriptors
4. Implement the async `run` method, yielding `(pin_name, value)` tuples
5. Generate the block ID using `uuid.uuid4()` -- must be a valid UUID
6. The block auto-registers via dynamic discovery (`load_all_blocks()`)
7. Run block tests: `poetry run pytest backend/blocks/test/test_block.py -xvs`

**Design consideration**: When creating blocks, think about how their inputs and outputs connect in the visual graph editor. Do the types compose well with other blocks?

Example block structure:
```python
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

class MyBlock(Block):
    class Input(BlockSchema):
        text: str = SchemaField(description="Input text to process")

    class Output(BlockSchema):
        result: str = SchemaField(description="Processed output")

    def __init__(self):
        super().__init__(
            id="<uuid4>",
            description="Does something useful",
            input_schema=MyBlock.Input,
            output_schema=MyBlock.Output,
            categories={BlockCategory.BASIC},
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        yield ("result", input_data.text.upper())
```

### Modifying the API

1. Update routes in `/backend/backend/server/routers/` (v1.py is the main router, v2/ for new features)
2. Add/update Pydantic models in the same directory or `/backend/backend/data/`
3. Write colocated tests (`*_test.py` beside the route file)
4. Run `poetry run test` to verify
5. Frontend API client auto-regenerates on next `pnpm dev` or `pnpm generate:api`

### Frontend Feature Development

1. Components go in `/frontend/src/components/` organized by domain
2. Use existing UI components from `/frontend/src/components/ui/`
3. Add Storybook stories for new components
4. Test with Playwright if user-facing

**API client generation (Frontend):**

```bash
# Regenerate API client from OpenAPI spec
pnpm generate:api:force
```

Uses Orval to generate two clients: `autogpt_api_client` and `autogpt_zod_schema`

**Adding a graph template:**

1. Create workflow JSON in `/graph_templates/`
2. Follow existing template patterns (Discord bots, Medium blogger, etc.)
3. Templates are importable in the agent builder

## Testing Details

### Backend Testing

- **Framework:** pytest with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- **Test location:** Colocated with source -- `module.py` -> `module_test.py`
- **Snapshot testing:** pytest-snapshot for API response validation
  - Snapshots stored in `snapshots/` directories
  - Update: `poetry run pytest path/to/test.py --snapshot-update`
  - Always review diffs before committing snapshots
  - Exclude dynamic data (timestamps, IDs) from snapshots
- **Block tests:** Parametric validation of all block types via `test_available_blocks`
- **Test isolation:** Uses Docker-based test database (`run_tests.py`)
- **Mocking:** pytest-mock for external services; FastAPI `dependency_overrides` for auth

### Frontend Testing

- **Framework:** Playwright 1.54 for E2E tests
- **Config:** `playwright.config.ts` -- Chromium only, parallel execution
- **Pattern:** Page Object Model with reusable page classes
- **Location:** `/frontend/src/tests/*.spec.ts` (20+ test suites)
- **Global setup:** Creates test users dynamically
- **Component testing:** Storybook 9.1 with MSW for API mocking
- **Visual regression:** Chromatic integration
- **Accessibility:** axe-playwright for a11y testing

## Code Style

### Python

- **Formatter:** Black (default settings)
- **Import sorting:** isort with Black profile
- **Linting:** Ruff (target Python 3.10)
- **Type checking:** Pyright
- **Async:** Use `async def` for endpoints and block `run` methods
- **Dependencies:** Insert in alphabetical order in `pyproject.toml`

### TypeScript

- **Strict mode** enabled
- **ESLint:** extends `next/core-web-vitals`, `next/typescript`, `@tanstack/query/recommended`
- `react-hooks/exhaustive-deps`: OFF (rely on code review)
- `@typescript-eslint/no-explicit-any`: OFF (case-by-case)
- Unused vars with `_` prefix are allowed
- **Prettier:** with `prettier-plugin-tailwindcss` for class sorting
- **Path alias:** `@/*` -> `./src/*`

## Creating Pull Requests

- Target the `dev` branch (not `master`)
- Use conventional commit format for PR titles
- Fill out `.github/PULL_REQUEST_TEMPLATE.md` (Changes section + checklist)
- Keep out-of-scope changes under 20% of the PR
- For `data/*.py` changes, validate user ID checks or explain why not needed
- For new protected frontend routes, update `frontend/src/lib/supabase/middleware.ts`
- For configuration changes, update `.env.default` and `docker-compose.yml`
- Run pre-commit hooks before submitting

## Reviewing/Revising Pull Requests

```bash
# Get PR reviews
gh api /repos/Significant-Gravitas/AutoGPT/pulls/{pr_number}/reviews

# Get review comments
gh api /repos/Significant-Gravitas/AutoGPT/pulls/{pr_number}/reviews/{review_id}/comments

# Get PR-specific comments
gh api /repos/Significant-Gravitas/AutoGPT/issues/{pr_number}/comments

# Get PR diff
gh pr diff {pr_number}
```

## Conventional Commits

Use this format for commit messages and Pull Request titles:

**Types:**
- `feat`: Introduces a new feature
- `fix`: Patches a bug
- `refactor`: Code change that neither fixes a bug nor adds a feature; also applies to removing features
- `ci`: Changes to CI configuration
- `docs`: Documentation-only changes
- `dx`: Improvements to developer experience

**Scopes:**
- `platform`: Changes affecting both frontend and backend
- `frontend`: Frontend-only changes
- `backend`: Backend-only changes
- `blocks`: Block modifications or additions
- `infra`: Infrastructure changes

**Sub-scopes:**
- `backend/executor`, `backend/db`
- `frontend/builder` (includes block UI component changes)
- `frontend/library`, `frontend/marketplace`
- `infra/prod`

## Quick Setup

For new development environments, use the automated installers:

```bash
# Linux/macOS
./installer/setup-autogpt.sh

# Windows (run in PowerShell as Administrator)
.\installer\setup-autogpt.bat
```

These scripts handle prerequisite checking, repo setup, and service startup.

## Key Dependencies

### Backend
- `anthropic`, `openai`, `groq`, `ollama`: LLM providers
- `mem0ai`: Memory integration for agents
- `stripe`: Payment processing for credits
- `gcloud-aio-storage`: Google Cloud Storage async support
- `firecrawl-py`: Web crawling/scraping
- `exa-py`: Exa search integration
- `tiktoken`: Token counting for LLMs
- `aioclamd`: ClamAV virus scanning (async)
- `sentry-sdk`: Error monitoring (with FastAPI, OpenAI, Anthropic, LaunchDarkly plugins)
- `launchdarkly-server-sdk`: Feature flags
- `pinecone`: Vector database
- `replicate`: AI model hosting
- `moviepy`: Video processing
- `pandas`: Data manipulation

### Frontend
- `@xyflow/react`: Visual workflow editor
- `@tanstack/react-query`: Server state management
- `@tanstack/react-table`: Advanced table management
- `react-shepherd`: Product tours/onboarding guides
- `react-hook-form` + `zod`: Form handling and validation
- `framer-motion`: Animations
- `recharts`: Data visualization
- `sonner`: Toast notifications
- `cmdk`: Command palette
- `launchdarkly-react-client-sdk`: Feature flags
- `orval`: API client generation
- `@sentry/nextjs`: Error monitoring
