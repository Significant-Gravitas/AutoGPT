# AutoGPT Database Docker

This is a minimal Docker Compose setup for running PostgreSQL for the AutoGPT Platform.

## Usage

```bash
# Start the database
docker compose up -d

# Stop the database
docker compose down

# Destroy (remove volumes)
docker compose down -v --remove-orphans
```

## Configuration

The PostgreSQL database is configured with:
- Logical replication enabled (for Prisma)
- pgvector/pgvector:pg18 image (PostgreSQL 18 with pgvector extension for AI embeddings)
- Data persisted in `./volumes/db/data`

## Environment Variables

You can override the default configuration by setting environment variables:
- `POSTGRES_USER` - Database user (default: postgres)
- `POSTGRES_PASSWORD` - Database password (default: your-super-secret-and-long-postgres-password)
- `POSTGRES_DB` - Database name (default: postgres)
