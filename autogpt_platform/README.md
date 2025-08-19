# AutoGPT Platform

Welcome to the AutoGPT Platform - a powerful system for creating and running AI agents to solve business problems. This platform enables you to harness the power of artificial intelligence to automate tasks, analyze data, and generate insights for your organization.

## Getting Started

### Prerequisites

- Docker
- Docker Compose V2 (comes with Docker Desktop, or can be installed separately)

### Running the System

To run the AutoGPT Platform, follow these steps:

1. Clone this repository to your local machine and navigate to the `autogpt_platform` directory within the repository:

   ```
   git clone <https://github.com/Significant-Gravitas/AutoGPT.git | git@github.com:Significant-Gravitas/AutoGPT.git>
   cd AutoGPT/autogpt_platform
   ```

2. Run the following command:

   ```
   cp .env.default .env
   ```

   This command will copy the `.env.default` file to `.env`. You can modify the `.env` file to add your own environment variables.

3. Run the following command:

   ```
   docker compose up -d
   ```

   This command will start all the necessary backend services defined in the `docker-compose.yml` file in detached mode.

4. After all the services are in ready state, open your browser and navigate to `http://localhost:3000` to access the AutoGPT Platform frontend.

### Docker Compose Commands

Here are some useful Docker Compose commands for managing your AutoGPT Platform:

- `docker compose up -d`: Start the services in detached mode.
- `docker compose stop`: Stop the running services without removing them.
- `docker compose rm`: Remove stopped service containers.
- `docker compose build`: Build or rebuild services.
- `docker compose down`: Stop and remove containers, networks, and volumes.
- `docker compose watch`: Watch for changes in your services and automatically update them.

### Sample Scenarios

Here are some common scenarios where you might use multiple Docker Compose commands:

1. Updating and restarting a specific service:

   ```
   docker compose build api_srv
   docker compose up -d --no-deps api_srv
   ```

   This rebuilds the `api_srv` service and restarts it without affecting other services.

2. Viewing logs for troubleshooting:

   ```
   docker compose logs -f api_srv ws_srv
   ```

   This shows and follows the logs for both `api_srv` and `ws_srv` services.

3. Scaling a service for increased load:

   ```
   docker compose up -d --scale executor=3
   ```

   This scales the `executor` service to 3 instances to handle increased load.

4. Stopping the entire system for maintenance:

   ```
   docker compose stop
   docker compose rm -f
   docker compose pull
   docker compose up -d
   ```

   This stops all services, removes containers, pulls the latest images, and restarts the system.

5. Developing with live updates:

   ```
   docker compose watch
   ```

   This watches for changes in your code and automatically updates the relevant services.

6. Checking the status of services:
   ```
   docker compose ps
   ```
   This shows the current status of all services defined in your docker-compose.yml file.

These scenarios demonstrate how to use Docker Compose commands in combination to manage your AutoGPT Platform effectively.

### Persisting Data

To persist data for PostgreSQL and Redis, you can modify the `docker-compose.yml` file to add volumes. Here's how:

1. Open the `docker-compose.yml` file in a text editor.
2. Add volume configurations for PostgreSQL and Redis services:

   ```yaml
   services:
     postgres:
       # ... other configurations ...
       volumes:
         - postgres_data:/var/lib/postgresql/data

     redis:
       # ... other configurations ...
       volumes:
         - redis_data:/data

   volumes:
     postgres_data:
     redis_data:
   ```

3. Save the file and run `docker compose up -d` to apply the changes.

This configuration will create named volumes for PostgreSQL and Redis, ensuring that your data persists across container restarts.

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
