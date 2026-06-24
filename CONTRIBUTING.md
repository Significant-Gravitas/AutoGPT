# Contributing to AutoGPT

Thanks for your interest in contributing! This guide covers everything you need to get started.

## Repository Structure

```
AutoGPT/
├── autogpt_platform/        # Main platform (Polyform Shield License)
│   ├── backend/             # Python FastAPI server
│   ├── frontend/            # Next.js React application
│   ├── autogpt_libs/        # Shared Python utilities
│   └── docker-compose.yml   # Development stack
├── classic/                 # Legacy agent system (MIT License)
└── docs/                    # Documentation (gitbook branch)
```

## Prerequisites

- **Python** 3.11 (managed by Poetry)
- **Node.js** 21+ with pnpm
- **Docker** and Docker Compose
- **Poetry** for Python dependency management

## Getting Started

### 1. Start Infrastructure Services

All development requires the Docker services to be running:

```bash
cd autogpt_platform
docker compose --profile local up deps --build --detach
```

This starts PostgreSQL, Redis, RabbitMQ, and ClamAV.

### 2. Backend Setup

```bash
cd autogpt_platform/backend
poetry install
poetry run prisma migrate dev
poetry run prisma generate
```

Run the backend server:

```bash
poetry run serve  # Starts on port 8000
```

### 3. Frontend Setup

```bash
cd autogpt_platform/frontend
pnpm install
pnpm dev  # Starts on port 3000
```

## Development

### Backend Commands

```bash
cd autogpt_platform/backend
poetry run format          # Format code (Black + isort) — run this first
poetry run lint            # Lint code (ruff)
poetry run test            # Run all tests (~5 min)
poetry run pytest path/to/test.py -xvs  # Run specific test
```

### Frontend Commands

```bash
cd autogpt_platform/frontend
pnpm format               # Format and lint
pnpm dev                  # Development server
pnpm build                # Production build
pnpm test                 # Playwright E2E tests (requires dev server running)
pnpm storybook            # Component development server
pnpm generate:api         # Regenerate API client from OpenAPI spec
```

### Testing

**Backend:**
- All block tests: `poetry run pytest backend/blocks/test/test_block.py -xvs`
- Single block: `poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[BlockName]' -xvs`
- Snapshot tests: Use `--snapshot-update` when output changes, always review with `git diff`

**Frontend:**
- E2E tests require `pnpm dev` running
- Component tests use Storybook

### Adding a New Block

1. Create a file in `autogpt_platform/backend/backend/blocks/`
2. Inherit from `Block` base class with input/output schemas
3. Implement the `run` method with proper error handling
4. Generate a unique block ID with `uuid.uuid4()`
5. Write tests alongside the block
6. Validate with `poetry run pytest backend/blocks/test/test_block.py -xvs`

See the [Block SDK Guide](docs/platform/block-sdk-guide.md) for full details.

### Modifying the API

1. Update routes in `autogpt_platform/backend/backend/server/routers/`
2. Add/update Pydantic models in the same directory
3. Write tests alongside the route file
4. Run `poetry run test` to verify
5. If you change the schema, run `poetry run prisma migrate dev`

## Pull Request Process

### Conventional Commits

Use this format for commit messages and PR titles:

```
type(scope): description
```

**Types:** `feat`, `fix`, `refactor`, `ci`, `docs`, `dx`

**Scopes:** `platform`, `frontend`, `backend`, `blocks`, `infra`

**Subscope examples:** `backend/executor`, `frontend/builder`, `infra/prod`

### Branch Naming

Use descriptive branch names, e.g. `feature/add-new-block`, `fix/login-redirect`.

### Base Branch

- **Code changes:** Target the `dev` branch
- **Documentation changes:** Target the `gitbook` branch

### Before Submitting

1. Run `poetry run format` (backend) and `pnpm format` (frontend)
2. Ensure tests pass in modified areas
3. Fill out the PR template in `.github/PULL_REQUEST_TEMPLATE.md`
4. For changes to `data/*.py`, validate user ID checks
5. If adding protected frontend routes, update `frontend/lib/supabase/middleware.ts`

## Contributing to Documentation

Documentation lives in `docs/` on the **gitbook** branch. To contribute:

1. Check out the `gitbook` branch
2. Edit or add Markdown files in `docs/`
3. Open a PR targeting the `gitbook` branch

## Licensing

- **`autogpt_platform/`** — [Polyform Shield License](autogpt_platform/Contributor%20License%20Agreement%20(CLA).md). By submitting a PR to this folder you agree to the CLA.
- **All other folders** — MIT License.

## Additional Resources

- **Frontend Contributing Guide:** [autogpt_platform/frontend/CONTRIBUTING.md](autogpt_platform/frontend/CONTRIBUTING.md) — detailed frontend patterns and conventions
- **Block SDK Guide:** [docs/platform/block-sdk-guide.md](docs/platform/block-sdk-guide.md)
- **Discord:** [discord.gg/autogpt](https://discord.gg/autogpt)

## About CLAUDE.md Files

You may notice `CLAUDE.md` files in the repository. These are AI-optimized reference documents used by coding assistants. This CONTRIBUTING.md is the canonical human-readable guide — the CLAUDE.md files supplement it with machine-parseable context.
