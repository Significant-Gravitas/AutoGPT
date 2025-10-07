.PHONY: start-core stop-core logs-core format lint

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
