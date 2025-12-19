.PHONY: start-core stop-core logs-core format lint migrate run-backend run-frontend load-store-agents

# Run just Supabase + Redis + RabbitMQ
start-core:
	docker compose up -d deps

# Stop core services
stop-core:
	docker compose stop deps

reset-db:
	rm -rf db/docker/volumes/db/data
	cd backend && poetry run prisma migrate deploy
	cd backend && poetry run prisma generate
	
# View logs for core services
logs-core:
	docker compose logs -f deps

# Run formatting and linting for backend and frontend
format:
	cd backend && poetry run format
	cd frontend && pnpm format
	cd frontend && pnpm lint

init-env:
	cp -n .env.default .env || true
	cd backend && cp -n .env.default .env || true
	cd frontend && cp -n .env.default .env || true


# Run migrations for backend
migrate:
	cd backend && poetry run prisma migrate deploy
	cd backend && poetry run prisma generate

run-backend:
	cd backend && poetry run app

run-frontend:
	cd frontend && pnpm dev

test-data:
	cd backend && poetry run python test/test_data_creator.py

load-store-agents:
	cd backend && poetry run load-store-agents

help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  start-core - Start just the core services (Supabase, Redis, RabbitMQ) in background"
	@echo "  stop-core - Stop the core services"
	@echo "  reset-db - Reset the database by deleting the volume"
	@echo "  logs-core - Tail the logs for core services"
	@echo "  format - Format & lint backend (Python) and frontend (TypeScript) code"
	@echo "  migrate - Run backend database migrations"
	@echo "  run-backend - Run the backend FastAPI server"
	@echo "  run-frontend - Run the frontend Next.js development server"
	@echo "  test-data - Run the test data creator"
	@echo "  load-store-agents - Load store agents from agents/ folder into test database"