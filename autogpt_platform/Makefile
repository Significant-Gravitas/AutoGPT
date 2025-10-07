.PHONY: core-services stop-core logs-core

# Run just Supabase + Redis + RabbitMQ
core-services:
	docker compose up -d kong auth db redis rabbitmq

# Stop core services
stop-core:
	docker compose stop kong auth db redis rabbitmq

# View logs for core services
logs-core:
	docker compose logs -f kong auth db redis rabbitmq

# Run formatting and linting for backend and frontend
format:
	cd backend && poetry run format
	cd frontend && pnpm format
	cd frontend && pnpm lint
