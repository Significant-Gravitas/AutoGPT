# GitHub Codespaces for AutoGPT Platform

This dev container provides a complete development environment for the AutoGPT Platform, optimized for PR reviews.

## 🚀 Quick Start

1. **Open in Codespaces:**
   - Go to the repository on GitHub
   - Click **Code** → **Codespaces** → **Create codespace on dev**
   - Or click the badge: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Significant-Gravitas/AutoGPT?quickstart=1)

2. **Wait for setup** (~60 seconds with prebuild, ~5-10 min without)

3. **Start the servers:**
   ```bash
   # Terminal 1
   make run-backend

   # Terminal 2  
   make run-frontend
   ```

4. **Start developing!**
   - Frontend: `http://localhost:3000`
   - Login with: `test123@gmail.com` / `testpassword123`

## 🏗️ Architecture

**Dependencies run in Docker** (cached by prebuild):
- PostgreSQL, Redis, RabbitMQ, Supabase Auth

**Backend & Frontend run natively** (not cached):
- This ensures you're always running the current branch's code
- Enables hot-reload during development
- VS Code debugger can attach directly

## 📍 Available Services

| Service | URL | Notes |
|---------|-----|-------|
| Frontend | http://localhost:3000 | Next.js app |
| REST API | http://localhost:8006 | FastAPI backend |
| WebSocket | ws://localhost:8001 | Real-time updates |
| Supabase | http://localhost:8000 | Auth & API gateway |
| Supabase Studio | http://localhost:5555 | Database admin |
| RabbitMQ | http://localhost:15672 | Queue management |

## 🔑 Test Accounts

| Email | Password | Role |
|-------|----------|------|
| test123@gmail.com | testpassword123 | Featured Creator |

The test account has:
- Pre-created agents and workflows
- Published store listings
- Active agent executions
- Reviews and ratings

## 🛠️ Development Commands

```bash
# Navigate to platform directory (terminal starts here by default)
cd autogpt_platform

# Start all services
docker compose up -d

# Or just core services (DB, Redis, RabbitMQ)
make start-core

# Run backend in dev mode (hot reload)
make run-backend

# Run frontend in dev mode (hot reload)
make run-frontend

# Run both backend and frontend
# (Use VS Code's "Full Stack" launch config for debugging)

# Format code
make format

# Run tests
make test-data        # Regenerate test data
poetry run test       # Backend tests (from backend/)
pnpm test:e2e         # E2E tests (from frontend/)
```

## 🐛 Debugging

### VS Code Launch Configs

> **Note:** Launch and task configs are in `.devcontainer/vscode-templates/`. 
> To use them locally, copy to `.vscode/`:
> ```bash
> cp .devcontainer/vscode-templates/*.json .vscode/
> ```
> In Codespaces, core settings are auto-applied via devcontainer.json.

Press `F5` or use the Run and Debug panel:

- **Backend: Debug FastAPI** - Debug the REST API server
- **Backend: Debug Executor** - Debug the agent executor
- **Frontend: Debug Next.js** - Debug with browser DevTools
- **Full Stack: Backend + Frontend** - Debug both simultaneously
- **Tests: Run E2E Tests** - Run Playwright tests

### VS Code Tasks

Press `Ctrl+Shift+P` → "Tasks: Run Task":

- Start/Stop All Services
- Run Migrations
- Seed Test Data
- View Docker Logs
- Reset Database

## 📁 Project Structure

```text
autogpt_platform/           # This folder
├── .devcontainer/          # Codespaces/devcontainer config
├── .vscode/                # VS Code settings
├── backend/                # Python FastAPI backend
│   ├── backend/            # Application code
│   ├── test/               # Test files + data seeders
│   └── migrations/         # Prisma migrations
├── frontend/               # Next.js frontend
│   ├── src/                # Application code
│   └── e2e/                # Playwright E2E tests
├── db/                     # Supabase configuration
├── docker-compose.yml      # Service orchestration
└── Makefile                # Common commands
```

## 🔧 Troubleshooting

### Services not starting?
```bash
# Check service status
docker compose ps

# View logs
docker compose logs -f

# Restart everything
docker compose down && docker compose up -d
```

### Database issues?
```bash
# Reset database (destroys all data)
make reset-db

# Re-run migrations
make migrate

# Re-seed test data
make test-data
```

### Port already in use?
```bash
# Check what's using the port
lsof -i :3000

# Kill process (if safe)
kill -9 <PID>
```

### Can't login?
- Ensure all services are running: `docker compose ps`
- Check auth service: `docker compose logs auth`
- Try seeding data again: `make test-data`

## 📝 Making Changes

### Backend Changes
1. Edit files in `backend/backend/`
2. If using `make run-backend`, changes auto-reload
3. Run `poetry run format` before committing

### Frontend Changes
1. Edit files in `frontend/src/`
2. If using `make run-frontend`, changes auto-reload
3. Run `pnpm format` before committing

### Database Schema Changes
1. Edit `backend/schema.prisma`
2. Run `poetry run prisma migrate dev --name your_migration`
3. Run `poetry run prisma generate`

## 🔒 Environment Variables

Default environment variables are configured for local development. For production secrets, use GitHub Codespaces Secrets:

1. Go to GitHub Settings → Codespaces → Secrets
2. Add secrets with names matching `.env` variables
3. They'll be automatically available in your codespace

## 📚 More Resources

- [AutoGPT Platform Docs](https://docs.agpt.co)
- [Codespaces Documentation](https://docs.github.com/en/codespaces)
- [Dev Containers Spec](https://containers.dev)
