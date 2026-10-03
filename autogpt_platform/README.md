# AutoGPT Platform

Welcome to the AutoGPT Platform - a powerful system for creating and running AI agents to solve business problems. This platform enables you to harness the power of artificial intelligence to automate tasks, analyze data, and generate insights for your organization.

> **Just want to run AutoGPT?** The quickest way is the single-container image on Docker Hub, [`significantgravitas/autogpt`](https://hub.docker.com/r/significantgravitas/autogpt). See the [self-hosting guide](https://docs.agpt.co/platform/self-hosting/getting-started). The steps below build and run every service from this checkout with Docker Compose.

## Getting Started

### Prerequisites

- Docker
- Docker Compose V2 (comes with Docker Desktop, or can be installed separately)
- `make` and Python 3 (used to generate your local secrets)

### Running the System

To run the AutoGPT Platform, follow these steps:

1. Clone this repository to your local machine and navigate to the `autogpt_platform` directory within the repository:

   ```
   git clone https://github.com/Significant-Gravitas/AutoGPT.git
   cd AutoGPT/autogpt_platform
   ```

2. Run the following command:

   ```
   make init-env
   ```

   This command copies each `.env.default` file to `.env` (in `autogpt_platform`, `backend` and `frontend`) where no `.env` exists yet, and generates the secrets those files leave blank: `ENCRYPTION_KEY`, `UNSUBSCRIBE_SECRET_KEY` and `BETTER_AUTH_SECRET`. The backend refuses to start without `ENCRYPTION_KEY`. It never overwrites an existing file or value, so it is safe to re-run. You can then modify the `.env` files to add your own environment variables.

   Without `make`, run `installer/setup-autogpt.sh` (Linux/macOS) or `installer\setup-autogpt.bat` (Windows) instead. They do the same and then start the services.

3. Run the following command:

   ```
   docker compose up -d
   ```

   This command will start all the necessary backend services defined in the `docker-compose.yml` file in detached mode.

4. After all the services are in ready state, open your browser and navigate to `http://localhost:3000` to access the AutoGPT Platform frontend.

### Running Just Core services

You can now run the following to enable just the core services.

```
# For help
make help

# Run the services the backend needs (Postgres, Redis, RabbitMQ, ClamAV, FalkorDB) and apply migrations
make start-core

# Stop every running service in the stack
make stop-core

# View logs from core services 
make logs-core

# Run formatting and linting for backend and frontend
make format

# Run migrations for backend database
make migrate

# Run backend server
make run-backend

# Run frontend development server
make run-frontend

```

### Docker Compose Commands

Here are some useful Docker Compose commands for managing your AutoGPT Platform:

- `docker compose up -d`: Start the services in detached mode.
- `docker compose stop`: Stop the running services without removing them.
- `docker compose rm`: Remove stopped service containers.
- `docker compose build`: Build or rebuild services.
- `docker compose down`: Stop and remove containers and networks. Add `-v` to also delete the named volumes (workspace files, marketplace media, FalkorDB memory, and the ClamAV database); the database in `data/db/data` survives both.
- `docker compose watch`: Watch for changes in your services and automatically update them.

### Sample Scenarios

Here are some common scenarios where you might use multiple Docker Compose commands:

1. Updating and restarting a specific service:

   ```
   docker compose build rest_server
   docker compose up -d --no-deps rest_server
   ```

   This rebuilds the `rest_server` service and restarts it without affecting other services.

2. Viewing logs for troubleshooting:

   ```
   docker compose logs -f rest_server websocket_server
   ```

   This shows and follows the logs for both `rest_server` and `websocket_server` services.

3. Stopping the entire system for maintenance:

   ```
   docker compose stop
   docker compose rm -f
   docker compose pull --ignore-buildable
   docker compose up -d --build
   ```

   This stops all services, removes containers, pulls the latest images, and
   restarts the system. `--ignore-buildable` skips the services this repo
   builds from source; without it `pull` tries to fetch them from a registry
   they were never published to and fails.

4. Developing with live updates:

   ```
   docker compose watch
   ```

   This watches for changes in your code and automatically updates the relevant services.

5. Checking the status of services:
   ```
   docker compose ps
   ```
   This shows the current status of all services defined in your docker-compose.yml file.

These scenarios demonstrate how to use Docker Compose commands in combination to manage your AutoGPT Platform effectively.

### Persisting Data

Your data already persists across restarts:

- PostgreSQL keeps its data in `autogpt_platform/data/db/data` on the host.
- Workspace files, marketplace media, and FalkorDB memory use the named volumes `workspace-data`, `store-media-data`, and `falkordb_data`.
- The three Redis nodes are a cache and are not persisted; do not add volumes to them.

Back up `data/db/data` (with the stack stopped) together with your `.env` files: `backend/.env` holds the `ENCRYPTION_KEY` your stored integration credentials are encrypted with.

### API Client Generation

The platform includes scripts for generating and managing the API client:

- `pnpm fetch:openapi`: Fetches the OpenAPI specification from the backend service (requires backend to be running on port 8006)
- `pnpm generate:api-client`: Generates the TypeScript API client from the OpenAPI specification using Orval
- `pnpm generate:api`: Runs both fetch and generate commands in sequence

#### Manual API Client Updates

If you need to update the API client after making changes to the backend API:

1. Ensure the backend services are running:

   ```
   docker compose up -d
   ```

2. Generate the updated API client:
   ```
   pnpm generate:api
   ```

This will fetch the latest OpenAPI specification and regenerate the TypeScript client code.
