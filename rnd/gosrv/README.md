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
