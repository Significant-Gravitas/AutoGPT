.PHONY: start-core stop-core logs-core format lint

# Run just Supabase + Redis + RabbitMQ
start-core:
	docker compose up -d kong auth db redis rabbitmq studio

# Stop core services
stop-core:
	docker compose stop kong auth db redis rabbitmq studio

# View logs for core services
logs-core:
	docker compose logs -f kong auth db redis rabbitmq studio

# Run formatting and linting for backend and frontend
format:
	cd backend && poetry run format
	cd frontend && pnpm format
	cd frontend && pnpm lint
