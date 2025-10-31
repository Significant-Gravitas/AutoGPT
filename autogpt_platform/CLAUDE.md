# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

AutoGPT Platform is a monorepo containing:

- **Backend** (`/backend`): Python FastAPI server with async support
- **Frontend** (`/frontend`): Next.js React application
- **Shared Libraries** (`/autogpt_libs`): Common Python utilities

## Essential Commands

### Backend Development

```bash
# Install dependencies
cd backend && poetry install

# Run database migrations
poetry run prisma migrate dev

# Start all services (database, redis, rabbitmq, clamav)
docker compose up -d

# Run the backend server
poetry run serve

# Run tests
poetry run test

# Run specific test
poetry run pytest path/to/test_file.py::test_function_name

# Run block tests (tests that validate all blocks work correctly)
poetry run pytest backend/blocks/test/test_block.py -xvs

# Run tests for a specific block (e.g., GetCurrentTimeBlock)
poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[GetCurrentTimeBlock]' -xvs

# Lint and format
# prefer format if you want to just "fix" it and only get the errors that can't be autofixed
poetry run format  # Black + isort
poetry run lint    # ruff
```

More details can be found in TESTING.md

#### Creating/Updating Snapshots

When you first write a test or when the expected output changes:

```bash
poetry run pytest path/to/test.py --snapshot-update
```

⚠️ **Important**: Always review snapshot changes before committing! Use `git diff` to verify the changes are expected.

### Frontend Development

```bash
# Install dependencies
cd frontend && pnpm i

# Generate API client from OpenAPI spec
pnpm generate:api

# Start development server
pnpm dev

# Run E2E tests
pnpm test

# Run Storybook for component development
pnpm storybook

# Build production
pnpm build

# Format and lint
pnpm format

# Type checking
pnpm types
```

**📖 Complete Guide**: See `/frontend/CONTRIBUTING.md` and `/frontend/.cursorrules` for comprehensive frontend patterns.

**Key Frontend Conventions:**

- Separate render logic from data/behavior in components
- Use generated API hooks from `@/app/api/__generated__/endpoints/`
- Use function declarations (not arrow functions) for components/handlers
- Use design system components from `src/components/` (atoms, molecules, organisms)
- Only use Phosphor Icons
- Never use `src/components/__legacy__/*` or deprecated `BackendAPI`

## Architecture Overview

### Backend Architecture

- **API Layer**: FastAPI with REST and WebSocket endpoints
- **Database**: PostgreSQL with Prisma ORM, includes pgvector for embeddings
- **Queue System**: RabbitMQ for async task processing
- **Execution Engine**: Separate executor service processes agent workflows
- **Authentication**: JWT-based with Supabase integration
- **Security**: Cache protection middleware prevents sensitive data caching in browsers/proxies

### Frontend Architecture

- **Framework**: Next.js 15 App Router (client-first approach)
- **Data Fetching**: Type-safe generated API hooks via Orval + React Query
- **State Management**: React Query for server state, co-located UI state in components/hooks
- **Component Structure**: Separate render logic (`.tsx`) from business logic (`use*.ts` hooks)
- **Workflow Builder**: Visual graph editor using @xyflow/react
- **UI Components**: shadcn/ui (Radix UI primitives) with Tailwind CSS styling
- **Icons**: Phosphor Icons only
- **Feature Flags**: LaunchDarkly integration
- **Error Handling**: ErrorCard for render errors, toast for mutations, Sentry for exceptions
- **Testing**: Playwright for E2E, Storybook for component development

### Key Concepts

1. **Agent Graphs**: Workflow definitions stored as JSON, executed by the backend
2. **Blocks**: Reusable components in `/backend/blocks/` that perform specific tasks
3. **Integrations**: OAuth and API connections stored per user
4. **Store**: Marketplace for sharing agent templates
5. **Virus Scanning**: ClamAV integration for file upload security

### Testing Approach

- Backend uses pytest with snapshot testing for API responses
- Test files are colocated with source files (`*_test.py`)
- Frontend uses Playwright for E2E tests
- Component testing via Storybook

### Database Schema

Key models (defined in `/backend/schema.prisma`):

- `User`: Authentication and profile data
- `AgentGraph`: Workflow definitions with version control
- `AgentGraphExecution`: Execution history and results
- `AgentNode`: Individual nodes in a workflow
- `StoreListing`: Marketplace listings for sharing agents

### Environment Configuration

#### Configuration Files

- **Backend**: `/backend/.env.default` (defaults) → `/backend/.env` (user overrides)
- **Frontend**: `/frontend/.env.default` (defaults) → `/frontend/.env` (user overrides)
- **Platform**: `/.env.default` (Supabase/shared defaults) → `/.env` (user overrides)

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

### Common Development Tasks

**Adding a new block:**

Follow the comprehensive [Block SDK Guide](../../../docs/content/platform/block-sdk-guide.md) which covers:

- Provider configuration with `ProviderBuilder`
- Block schema definition
- Authentication (API keys, OAuth, webhooks)
- Testing and validation
- File organization

Quick steps:

1. Create new file in `/backend/backend/blocks/`
2. Configure provider using `ProviderBuilder` in `_config.py`
3. Inherit from `Block` base class
4. Define input/output schemas using `BlockSchema`
5. Implement async `run` method
6. Generate unique block ID using `uuid.uuid4()`
7. Test with `poetry run pytest backend/blocks/test/test_block.py`

Note: when making many new blocks analyze the interfaces for each of these blocks and picture if they would go well together in a graph based editor or would they struggle to connect productively?
ex: do the inputs and outputs tie well together?

If you get any pushback or hit complex block conditions check the new_blocks guide in the docs.

**Modifying the API:**

1. Update route in `/backend/backend/server/routers/`
2. Add/update Pydantic models in same directory
3. Write tests alongside the route file
4. Run `poetry run test` to verify

**Frontend feature development:**

See `/frontend/CONTRIBUTING.md` for complete patterns. Quick reference:

1. **Pages**: Create in `src/app/(platform)/feature-name/page.tsx`
   - Add `usePageName.ts` hook for logic
   - Put sub-components in local `components/` folder
2. **Components**: Structure as `ComponentName/ComponentName.tsx` + `useComponentName.ts` + `helpers.ts`
   - Use design system components from `src/components/` (atoms, molecules, organisms)
   - Never use `src/components/__legacy__/*`
3. **Data fetching**: Use generated API hooks from `@/app/api/__generated__/endpoints/`
   - Regenerate with `pnpm generate:api`
   - Pattern: `use{Method}{Version}{OperationName}`
4. **Styling**: Tailwind CSS only, use design tokens, Phosphor Icons only
5. **Testing**: Add Storybook stories for new components, Playwright for E2E
6. **Code conventions**: Function declarations (not arrow functions) for components/handlers

### Security Implementation

**Cache Protection Middleware:**

- Located in `/backend/backend/server/middleware/security.py`
- Default behavior: Disables caching for ALL endpoints with `Cache-Control: no-store, no-cache, must-revalidate, private`
- Uses an allow list approach - only explicitly permitted paths can be cached
- Cacheable paths include: static assets (`/static/*`, `/_next/static/*`), health checks, public store pages, documentation
- Prevents sensitive data (auth tokens, API keys, user data) from being cached by browsers/proxies
- To allow caching for a new endpoint, add it to `CACHEABLE_PATHS` in the middleware
- Applied to both main API server and external API applications

### Creating Pull Requests

- Create the PR aginst the `dev` branch of the repository.
- Ensure the branch name is descriptive (e.g., `feature/add-new-block`)/
- Use conventional commit messages (see below)/
- Fill out the .github/PULL_REQUEST_TEMPLATE.md template as the PR description/
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
