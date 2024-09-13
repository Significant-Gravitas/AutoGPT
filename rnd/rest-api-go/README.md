# Market API

This project is a Go-based API for a marketplace application. It provides endpoints for managing agents, handling user authentication, and performing administrative tasks.

## Project Structure

The project is organized into several packages:

- `config`: Handles configuration loading and management
- `docs`: Contains the Swagger documentation
- `database`: Contains database migrations and interaction logic
- `handlers`: Implements HTTP request handlers
- `middleware`: Contains middleware functions for the API
- `models`: Defines data structures used throughout the application
- `utils`: Provides utility functions

## Prerequisites

- Go 1.16 or later
- PostgreSQL
- [golang-migrate](https://github.com/golang-migrate/migrate)

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   go mod tidy
   ```
3. Set up the database:
   - Create a PostgreSQL database
   - Update the `DatabaseURL` in your configuration file

4. Run database migrations:
   ```
   migrate -source file://database/migrations -database "postgresql://agpt_user:pass123@localhost:5432/apgt_marketplace?sslmode=disable" up
   ```

## Running the Application

To run the application in development mode with hot reloading:

```
air
```

For production, build and run the binary:

```
go build -o market-api
./market-api
```

## Testing

Run tests with coverage:
```
go test -cover ./...
```

## Code Formatting

Format the code using:

```
gofmt -w .
```

## Database Migrations

Create a new migration:

```
migrate create -ext sql -dir database/migrations -seq <migration_name>
```

Apply migrations:

```
migrate -source file://database/migrations -database "postgresql://user:password@localhost:5432/dbname?sslmode=disable" up
```

Revert the last migration:

```
migrate -source file://database/migrations -database "postgresql://user:password@localhost:5432/dbname?sslmode=disable" down 1
```

## API Endpoints

The API provides various endpoints for agent management, user authentication, and administrative tasks. Some key endpoints include:

- `/api/agents`: Get list of agents
- `/api/agents/:agent_id`: Get agent details
- `/api/agents/submit`: Submit a new agent
- `/api/admin/*`: Various administrative endpoints (requires admin authentication)

Refer to the `main.go` file for a complete list of endpoints and their corresponding handlers.


# Swagger Documentation

This project uses `gin-swagger` and `Swaggo` tools for automatic generation of API documentation in OpenAPI (Swagger) format. The documentation is based on comments added to the code using Swagger annotations.

To view and interact with the generated Swagger documentation, follow these steps:

1. Run your Gin server.
2. Access the Swagger UI by navigating to `http://localhost:8015/docs/index.html` in your web browser.

Alternatively, you can view the raw OpenAPI specification at `http://localhost:8015/docs/doc.json`.

## Regenerating Swagger Documentation

If you make changes to your codebase and want to regenerate the Swagger documentation, follow these steps:

1. Run the `swag init` command in your project directory to create a new `docs.go` file (or update an existing one) with Swagger documentation comments based on your code:
```bash
swag init -g main.go
```
Replace `main.go` with the name of your main Go source file.

3. Run your Gin server, and access the updated Swagger UI at `http://localhost:8015/docs/index.html`. You should see your documentation reflecting the latest changes in your codebase.