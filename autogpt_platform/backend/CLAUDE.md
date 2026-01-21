# CLAUDE.md - Backend

This file provides guidance to Claude Code when working with the backend.

## Essential Commands

To run something with Python package dependencies you MUST use `poetry run ...`.

```bash
# Install dependencies
cd backend && poetry install

# Run database migrations
poetry run prisma migrate dev

# Start all services (database, redis, rabbitmq, clamav)
docker compose up -d

# Run the backend as a whole
poetry run app

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

More details can be found in @TESTING.md

### Creating/Updating Snapshots

When you first write a test or when the expected output changes:

```bash
poetry run pytest path/to/test.py --snapshot-update
```

⚠️ **Important**: Always review snapshot changes before committing! Use `git diff` to verify the changes are expected.

## Architecture

- **API Layer**: FastAPI with REST and WebSocket endpoints
- **Database**: PostgreSQL with Prisma ORM, includes pgvector for embeddings
- **Queue System**: RabbitMQ for async task processing
- **Execution Engine**: Separate executor service processes agent workflows
- **Authentication**: JWT-based with Supabase integration
- **Security**: Cache protection middleware prevents sensitive data caching in browsers/proxies

## Testing Approach

- Uses pytest with snapshot testing for API responses
- Test files are colocated with source files (`*_test.py`)

## Database Schema

Key models (defined in `schema.prisma`):

- `User`: Authentication and profile data
- `AgentGraph`: Workflow definitions with version control
- `AgentGraphExecution`: Execution history and results
- `AgentNode`: Individual nodes in a workflow
- `StoreListing`: Marketplace listings for sharing agents

## Environment Configuration

- **Backend**: `.env.default` (defaults) → `.env` (user overrides)

## Common Development Tasks

### Adding a new block

Follow the comprehensive [Block SDK Guide](@../../docs/content/platform/block-sdk-guide.md) which covers:

- Provider configuration with `ProviderBuilder`
- Block schema definition
- Authentication (API keys, OAuth, webhooks)
- Testing and validation
- File organization

Quick steps:

1. Create new file in `backend/blocks/`
2. Configure provider using `ProviderBuilder` in `_config.py`
3. Inherit from `Block` base class
4. Define input/output schemas using `BlockSchema`
5. Implement async `run` method
6. Generate unique block ID using `uuid.uuid4()`
7. Test with `poetry run pytest backend/blocks/test/test_block.py`

Note: when making many new blocks analyze the interfaces for each of these blocks and picture if they would go well together in a graph based editor or would they struggle to connect productively?
ex: do the inputs and outputs tie well together?

If you get any pushback or hit complex block conditions check the new_blocks guide in the docs.

### Modifying the API

1. Update route in `backend/api/features/`
2. Add/update Pydantic models in same directory
3. Write tests alongside the route file
4. Run `poetry run test` to verify

## Security Implementation

### Cache Protection Middleware

- Located in `backend/server/middleware/security.py`
- Default behavior: Disables caching for ALL endpoints with `Cache-Control: no-store, no-cache, must-revalidate, private`
- Uses an allow list approach - only explicitly permitted paths can be cached
- Cacheable paths include: static assets (`static/*`, `_next/static/*`), health checks, public store pages, documentation
- Prevents sensitive data (auth tokens, API keys, user data) from being cached by browsers/proxies
- To allow caching for a new endpoint, add it to `CACHEABLE_PATHS` in the middleware
- Applied to both main API server and external API applications
