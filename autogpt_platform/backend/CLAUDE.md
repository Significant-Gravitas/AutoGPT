# CLAUDE.md - Backend

This file provides guidance to Claude Code when working with the backend.

## Essential Commands

To run something with Python package dependencies you MUST use `poetry run ...`.

```bash
# Install dependencies
poetry install

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

Note: when making many new blocks analyze the interfaces for each of these blocks and picture if they would go well together in a graph-based editor or would they struggle to connect productively?
ex: do the inputs and outputs tie well together?

If you get any pushback or hit complex block conditions check the new_blocks guide in the docs.

#### Handling files in blocks with `store_media_file()`

When blocks need to work with files (images, videos, documents), use `store_media_file()` from `backend.util.file`. The `return_format` parameter determines what you get back:

| Format | Use When | Returns |
|--------|----------|---------|
| `"for_local_processing"` | Processing with local tools (ffmpeg, MoviePy, PIL) | Local file path (e.g., `"image.png"`) |
| `"for_external_api"` | Sending content to external APIs (Replicate, OpenAI) | Data URI (e.g., `"data:image/png;base64,..."`) |
| `"for_block_output"` | Returning output from your block | Smart: `workspace://` in CoPilot, data URI in graphs |

**Examples:**

```python
# INPUT: Need to process file locally with ffmpeg
local_path = await store_media_file(
    file=input_data.video,
    execution_context=execution_context,
    return_format="for_local_processing",
)
# local_path = "video.mp4" - use with Path/ffmpeg/etc

# INPUT: Need to send to external API like Replicate
image_b64 = await store_media_file(
    file=input_data.image,
    execution_context=execution_context,
    return_format="for_external_api",
)
# image_b64 = "data:image/png;base64,iVBORw0..." - send to API

# OUTPUT: Returning result from block
result_url = await store_media_file(
    file=generated_image_url,
    execution_context=execution_context,
    return_format="for_block_output",
)
yield "image_url", result_url
# In CoPilot: result_url = "workspace://abc123"
# In graphs:  result_url = "data:image/png;base64,..."
```

**Key points:**

- `for_block_output` is the ONLY format that auto-adapts to execution context
- Always use `for_block_output` for block outputs unless you have a specific reason not to
- Never hardcode workspace checks - let `for_block_output` handle it

### Modifying the API

1. Update route in `backend/api/features/`
2. Add/update Pydantic models in same directory
3. Write tests alongside the route file
4. Run `poetry run test` to verify

## Security Implementation

### Cache Protection Middleware

- Located in `backend/api/middleware/security.py`
- Default behavior: Disables caching for ALL endpoints with `Cache-Control: no-store, no-cache, must-revalidate, private`
- Uses an allow list approach - only explicitly permitted paths can be cached
- Cacheable paths include: static assets (`static/*`, `_next/static/*`), health checks, public store pages, documentation
- Prevents sensitive data (auth tokens, API keys, user data) from being cached by browsers/proxies
- To allow caching for a new endpoint, add it to `CACHEABLE_PATHS` in the middleware
- Applied to both main API server and external API applications
