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
cd frontend && npm install

# Start development server
npm run dev

# Run E2E tests
npm run test

# Run Storybook for component development
npm run storybook

# Build production
npm run build

# Type checking
npm run type-check
```

## Architecture Overview

### Backend Architecture
- **API Layer**: FastAPI with REST and WebSocket endpoints
- **Database**: PostgreSQL with Prisma ORM, includes pgvector for embeddings
- **Queue System**: RabbitMQ for async task processing
- **Execution Engine**: Separate executor service processes agent workflows
- **Authentication**: JWT-based with Supabase integration
- **Security**: Cache protection middleware prevents sensitive data caching in browsers/proxies

### Frontend Architecture
- **Framework**: Next.js App Router with React Server Components
- **State Management**: React hooks + Supabase client for real-time updates
- **Workflow Builder**: Visual graph editor using @xyflow/react
- **UI Components**: Radix UI primitives with Tailwind CSS styling
- **Feature Flags**: LaunchDarkly integration

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
- Backend: `.env` file in `/backend`
- Frontend: `.env.local` file in `/frontend`
- Both require Supabase credentials and API keys for various services

### Common Development Tasks

**Adding a new block:**
1. Create new file in `/backend/backend/blocks/`
2. Inherit from `Block` base class
3. Define input/output schemas
4. Implement `run` method
5. Register in block registry
6. Generate the block uuid using `uuid.uuid4()`

**Modifying the API:**
1. Update route in `/backend/backend/server/routers/`
2. Add/update Pydantic models in same directory
3. Write tests alongside the route file
4. Run `poetry run test` to verify

**Frontend feature development:**
1. Components go in `/frontend/src/components/`
2. Use existing UI components from `/frontend/src/components/ui/`
3. Add Storybook stories for new components
4. Test with Playwright if user-facing

### Security Implementation

**Cache Protection Middleware:**
- Located in `/backend/backend/server/middleware/security.py`
- Default behavior: Disables caching for ALL endpoints with `Cache-Control: no-store, no-cache, must-revalidate, private`
- Uses an allow list approach - only explicitly permitted paths can be cached
- Cacheable paths include: static assets (`/static/*`, `/_next/static/*`), health checks, public store pages, documentation
- Prevents sensitive data (auth tokens, API keys, user data) from being cached by browsers/proxies
- To allow caching for a new endpoint, add it to `CACHEABLE_PATHS` in the middleware
- Applied to both main API server and external API applications