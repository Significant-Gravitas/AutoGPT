# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Architecture

The AutoGPT Platform is a microservice-based system for creating and running AI-powered agent workflows. It consists of three main components:

### Core Components

- **Backend** (`backend/`): Python FastAPI microservices with Redis, RabbitMQ, and PostgreSQL
- **Frontend** (`frontend/`): Next.js 14 application with TypeScript and Radix UI components  
- **Shared Libraries** (`autogpt_libs/`): Common Python utilities for auth, logging, rate limiting

### Service Architecture

The backend runs multiple services that communicate via Redis and RabbitMQ:

- **REST API Server** (port 8006-8007): Main HTTP API endpoints
- **WebSocket Server** (port 8001): Real-time communication for frontend
- **Executor** (port 8002): Handles workflow execution with block-based architecture
- **Scheduler** (port 8003): Manages scheduled agent runs
- **Database Manager**: Handles migrations and database connections
- **Notification Manager**: Email notifications and user alerts

### Data Model

- **AgentGraph**: Core workflow definition with nodes and links
- **AgentGraphExecution**: Runtime execution instances with status tracking
- **User**: Authentication via Supabase with credit system and integrations
- **Block**: Individual workflow components (400+ integrations supported)
- **LibraryAgent**: Reusable agent templates
- **StoreListing**: Marketplace for sharing agents

## Development Commands

### Backend Development
```bash
cd backend
poetry install
poetry run app          # All services
poetry run rest         # REST API only
poetry run ws           # WebSocket only
poetry run executor     # Executor only
poetry run scheduler    # Scheduler only
poetry run format       # Black + isort formatting
poetry run lint         # Ruff linting
poetry run test         # Run tests with Docker PostgreSQL
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev             # Development server (port 3000)
npm run build           # Production build
npm run lint            # ESLint + Prettier
npm run format          # Prettier only
npm run type-check      # TypeScript checking
npm run test            # Playwright E2E tests
npm run test-ui         # Playwright UI mode
npm run storybook       # Component development (port 6006)
```

### Docker Operations
```bash
docker compose up -d              # Start all backend services
docker compose stop               # Stop services
docker compose down               # Stop and remove containers
docker compose logs -f <service>  # View service logs
docker compose build <service>    # Rebuild specific service
```

### Database Management
```bash
cd backend
poetry run prisma migrate dev     # Apply migrations
poetry run prisma generate        # Generate Prisma client
poetry run prisma db push         # Push schema changes
```

## Code Architecture Patterns

### Block System
The core execution model uses a block-based architecture where each block represents an atomic operation:

- Blocks inherit from `backend.blocks.block.Block`
- Input/Output schemas defined using Pydantic models
- Blocks are auto-discovered and registered at runtime
- Each block has a unique UUID and category classification

### Data Layer
- **Prisma ORM** for PostgreSQL with Python async client
- **Repository pattern** in `backend/data/` modules
- **Pydantic models** for API serialization in `backend/data/model.py`
- **Database connection pooling** via `backend/data/db.py`

### API Architecture
- **FastAPI** with automatic OpenAPI generation
- **WebSocket support** for real-time execution updates
- **Supabase integration** for authentication and row-level security
- **Middleware** for auth, CORS, rate limiting in `autogpt_libs/`

### Frontend Architecture
- **Next.js App Router** with TypeScript
- **React Flow** for visual workflow builder (`@xyflow/react`)
- **Zustand/React Context** for state management
- **Radix UI** components with Tailwind CSS styling
- **Supabase client** for auth and real-time subscriptions

## Environment Setup

### Required Environment Variables

**Backend (.env)**:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_HOST` - Redis server for caching/sessions
- `RABBITMQ_HOST` - RabbitMQ for async messaging
- `SUPABASE_URL` + `SUPABASE_JWT_SECRET` - Authentication
- `ENABLE_AUTH=true` - Enable Supabase authentication

**Frontend (.env.local)**:
- `NEXT_PUBLIC_AGPT_SERVER_URL` - Backend REST API URL
- `NEXT_PUBLIC_AGPT_WS_SERVER_URL` - Backend WebSocket URL
- `NEXT_PUBLIC_SUPABASE_URL` + `NEXT_PUBLIC_SUPABASE_ANON_KEY` - Auth

### Integration Setup
The platform supports 400+ integrations requiring various API keys:
- **AI Providers**: OpenAI, Anthropic, Groq, Replicate
- **OAuth Providers**: GitHub, Google, Linear, Twitter, Todoist
- **Business Tools**: Stripe, HubSpot, Discord, Reddit

## Testing Strategy

### Backend Testing
- **pytest** with async support for unit/integration tests
- **Docker PostgreSQL** instance for database tests
- **Faker** for test data generation
- Run tests: `poetry run test`

### Frontend Testing  
- **Playwright** for end-to-end testing
- **Storybook** for component testing and documentation
- **TypeScript** strict mode for compile-time safety
- Run tests: `npm run test` or `npm run test-ui`

## Development Workflow

1. **Start backend services**: `docker compose up -d`
2. **Start frontend**: `cd frontend && npm run dev`
3. **Access application**: http://localhost:3000
4. **View Storybook**: http://localhost:6006
5. **Monitor logs**: `docker compose logs -f <service>`

### Code Quality
- **Backend**: Use `poetry run format` then `poetry run lint` before commits
- **Frontend**: Use `npm run format` then `npm run lint` before commits
- **Type checking**: Run `npm run type-check` for frontend TypeScript validation

### Database Changes
1. Edit `schema.prisma` file
2. Run `poetry run prisma migrate dev --name <migration_name>`
3. Commit both schema and migration files

## Performance Considerations

- **Executor scaling**: Use `docker compose up -d --scale executor=3` for high load
- **Redis caching**: Implemented for user sessions and API responses
- **Database indexing**: Key indexes on user_id, execution_id, created_at fields
- **Frontend optimization**: Next.js build includes automatic code splitting