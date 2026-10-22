# Advanced Setup

The advanced steps below are intended for people with sysadmin experience. If you are not comfortable with these steps, please refer to the [basic setup guide](../platform/getting-started.md).

## Introduction

For the advanced setup, first follow the [manual setup](getting-started.md#manual-setup) to get the server up and running. Once you have the server running, you can follow the steps below to configure the server for your specific needs. These steps apply to the Docker Compose setup; the [single-container image](single-container.md) is configured with the environment variables you pass to `docker run`.

## Configuration

### Setting config via environment variables

The server uses environment variables to store configs. Each part of the platform reads its own `.env` file inside `autogpt_platform/`:

- `backend/.env` for the backend services
- `frontend/.env` for the frontend
- `.env` for Docker Compose itself

The valid options are listed, with a comment for each, in the `.env.default` file next to each `.env`. `make init-env` creates any missing `.env` from its `.env.default` and generates the secrets that `.env.default` leaves blank:

```bash
cd autogpt_platform
make init-env
```

Then edit the `.env` files to change what you need. Do not copy `.env.default` over an existing `.env`: that erases your generated `ENCRYPTION_KEY`, and the integration credentials stored under it can no longer be read.

With Docker Compose, the `environment:` entries in `docker-compose.platform.yml` take precedence over the `.env` files. They connect the services to each other by their Compose names and set the database URLs, so change those settings in `docker-compose.platform.yml`. When you run the backend outside Docker, you can also set the environment variables directly in your shell. Refer to your operating system's documentation on how to set environment variables in the current session.

## Database selection

### PostgreSQL

We use PostgreSQL 15 with the pgvector extension as the database. Docker Compose runs it as the `db` service, keeps its data in `autogpt_platform/data/db/data`, and applies the Prisma migrations through the `migrate` service every time you run `docker compose up`.

To use your own PostgreSQL server instead:

1. Make sure the `vector` and `pg_trgm` extensions are available on it; the migrations create them.
2. Run `autogpt_platform/db/init/00-init.sql` against the database once. It creates the `platform` schema and an empty `auth.users` table that older migrations reference.
3. Point `DATABASE_URL` and `DIRECT_URL` at it, keeping `schema=platform` (in `docker-compose.platform.yml` under Docker, or in `backend/.env` when you run the backend directly), and point the frontend's `DATABASE_URL` at the same database.

The `migrate` service, or `make migrate` outside Docker, then applies the migrations.

## Redis

The platform uses Redis for caching, distributed locks, rate limits, and the streams that carry agent output to the browser.

The Docker Compose stack runs its own Redis Cluster: three `redis:7` shards on ports `17000` to `17002`, joined by a one-shot `redis-init` service. `make start-core` starts it along with the other dependencies.

### Using an external Redis cluster

The backend connects with a cluster client only, so the deployment must have cluster mode enabled. A standalone server, with cluster mode disabled, does not work.

The cluster also needs:

- **Redis 7.0 or later, or a compatible engine such as Valkey.** The backend uses sharded pub/sub (`SPUBLISH`, `SSUBSCRIBE`), which Redis 7.0 introduced, along with Streams and Lua scripts. It uses no Redis modules.
- **Shard addresses the backend can reach.** With `REDIS_USE_ANNOUNCED_ADDRESS=true`, the backend connects to each shard at the address the cluster announces, so those addresses must resolve from where the backend runs. Without it, the backend connects to every shard at `REDIS_HOST` and keeps only the announced port, which works only when every shard is reachable on that one host.
- **A private network between the backend and every shard.** The backend connects without TLS and does not send a username. It sends `REDIS_PASSWORD` when one is set, and a cluster without a password works too. Nothing on the connection is encrypted, so don't route it over a network you don't trust.

Point `REDIS_HOST` and `REDIS_PORT` at any node of the cluster, and set `REDIS_PASSWORD` if the cluster has one.

In the Compose stack, `REDIS_HOST`, `REDIS_PORT` and `REDIS_USE_ANNOUNCED_ADDRESS` are set in the `x-backend-env` block of `autogpt_platform/docker-compose.platform.yml`, which takes precedence over `backend/.env`, so change them there. `REDIS_PASSWORD` is read from `backend/.env`. The bundled shards still start, because the backend services wait for `redis-0` to be healthy.

## AutoGPT Agent Server Advanced set up

To run the backend server outside Docker with Poetry, follow [Backend Development](getting-started.md#backend-development).
