# Standalone Mode

This document describes how to run the AutoGPT platform in **standalone mode** without external dependencies like Redis, RabbitMQ, and Supabase.

## Overview

Standalone mode uses in-memory implementations for:
- **Redis**: In-memory key-value store and pub/sub messaging
- **RabbitMQ**: In-memory message queues and task distribution
- **Event Bus**: In-memory event distribution system

This is useful for:
- Development and testing without Docker
- Running in restricted environments
- Quick local demos
- CI/CD pipelines without external services

## Limitations

Standalone mode has these limitations:
- **No persistence**: All data is stored in memory and lost on restart
- **Single process**: Cannot scale across multiple workers/servers
- **No distributed locking**: Cluster coordination features are disabled
- **Database still required**: You still need PostgreSQL or need to configure SQLite
- **Authentication may need adjustment**: Supabase authentication won't work

## How to Enable

### Method 1: Environment Variable

Set the `STANDALONE_MODE` environment variable:

```bash
export STANDALONE_MODE=true
```

### Method 2: Configuration File

1. Copy the standalone environment template:
   ```bash
   cd autogpt_platform/backend
   cp .env.standalone .env
   ```

2. Edit `.env` and adjust any settings as needed

3. Add to the config (via environment or config.json):
   ```json
   {
     "standalone_mode": true
   }
   ```

### Method 3: Programmatic

In your code, you can check/set standalone mode:

```python
from backend.util.settings import Settings

settings = Settings()
settings.config.standalone_mode = True
```

## Running the Platform

Once standalone mode is enabled:

```bash
cd autogpt_platform/backend

# Install dependencies
poetry install

# Run migrations (if using SQLite, update schema first)
poetry run prisma migrate deploy

# Start the backend server
poetry run python -m backend.app
```

The platform will automatically use in-memory implementations when it detects:
- `STANDALONE_MODE=true` environment variable, OR
- `standalone_mode: true` in config settings

## Implementation Details

### Redis Client (`backend/data/redis_client.py`)

When standalone mode is active:
- Returns `InMemoryRedis()` instead of connecting to Redis
- Stores data in a Python dictionary (singleton)
- Supports basic operations: get, set, delete, incr, keys, etc.
- Pub/sub returns empty generators (use event bus instead)

### Event Bus (`backend/data/event_bus.py`)

When standalone mode is active:
- `RedisEventBus` uses `InMemorySyncEventBus`
- `AsyncRedisEventBus` uses `InMemoryAsyncEventBus`
- Events are distributed via Python queues
- Supports pattern subscriptions with wildcards

### RabbitMQ (`backend/data/rabbitmq.py`)

When standalone mode is active:
- `SyncRabbitMQ` uses `InMemorySyncRabbitMQ`
- `AsyncRabbitMQ` uses `InMemoryAsyncRabbitMQ`
- Messages are routed via Python queues
- Supports exchanges, queues, and routing keys
- Implements fanout, direct, and topic exchange types

## Troubleshooting

### "Connection refused" errors

If you still see connection errors:
1. Check that `STANDALONE_MODE=true` is set
2. Verify the setting with: `python -c "from backend.util.settings import Settings; print(Settings().config.standalone_mode)"`
3. Make sure you're not explicitly connecting to Redis/RabbitMQ elsewhere

### Database connection errors

Standalone mode does NOT replace the database. You need either:
- PostgreSQL running locally
- SQLite (requires schema changes)
- A remote database connection

### Authentication errors

Supabase authentication won't work in standalone mode. Options:
1. Disable authentication: `ENABLE_AUTH=false` in `.env`
2. Implement a mock auth provider
3. Use a different auth method

## Architecture

```
┌─────────────────────────────────────────┐
│          Standalone Mode                │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   InMemoryRedis (Singleton)      │  │
│  │   - Key-value store (dict)       │  │
│  │   - TTL not implemented          │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   InMemoryEventBus (Singleton)   │  │
│  │   - Pub/sub via queues           │  │
│  │   - Pattern matching support     │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   InMemoryMessageBroker          │  │
│  │   - Exchanges & queues           │  │
│  │   - Routing key matching         │  │
│  │   - Fanout/direct/topic types    │  │
│  └──────────────────────────────────┘  │
│                                         │
└─────────────────────────────────────────┘
```

## Testing

To test standalone mode:

```bash
# Set standalone mode
export STANDALONE_MODE=true

# Run tests
cd autogpt_platform/backend
poetry run pytest

# Or run a specific service
poetry run python -m backend.rest
```

## Production Use

**⚠️ Warning**: Standalone mode is NOT recommended for production use because:
- No data persistence
- No horizontal scaling
- No failover or redundancy
- Limited to single-process execution

For production, use the full Docker setup with real Redis, RabbitMQ, and PostgreSQL.

## Contributing

The standalone mode implementations are in:
- `backend/data/inmemory_redis.py` - In-memory Redis client
- `backend/data/inmemory_event_bus.py` - In-memory event bus
- `backend/data/inmemory_queue.py` - In-memory RabbitMQ

Feel free to improve these implementations or add missing features.
