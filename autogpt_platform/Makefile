.PHONY: start-core stop-core logs-core format lint, migrate

# Run just Supabase + Redis + RabbitMQ
start-core:
	docker compose up -d deps

# Stop core services
stop-core:
	docker compose stop deps

# View logs for core services
logs-core:
	docker compose logs -f deps

# Run formatting and linting for backend and frontend
format:
	cd backend && poetry run format
	cd frontend && pnpm format
	cd frontend && pnpm lint

# Run migrations for backend
migrate:
	cd backend && poetry run prisma migrate deploy
	cd backend && poetry run prisma generate

run-backend:
	cd backend && poetry run app

run-frontend:
	cd frontend && pnpm dev