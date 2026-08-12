# Advanced Setup

The advanced steps below are intended for people with sysadmin experience. If you are not comfortable with these steps, please refer to the [basic setup guide](../platform/getting-started.md).

## Introduction

For the advanced setup, first follow the [basic setup guide](../platform/getting-started.md) to get the server up and running. Once you have the server running, you can follow the steps below to configure the server for your specific needs.

## Configuration

### Setting config via environment variables

The server uses environment variables to store configs. You can set these environment variables in a `.env` file in the root of the project. The `.env` file should look like this:

```bash
# .env
KEY1=value1
KEY2=value2
```

The server will automatically load the `.env` file when it starts. You can also set the environment variables directly in your shell. Refer to your operating system's documentation on how to set environment variables in the current session.

The valid options are listed in `.env.default` in the root of the builder and server directories. You can copy the `.env.default` file to `.env` and modify the values as needed.

```bash
# Copy the .env.default file to .env
cp .env.default .env
```

### Secrets directory

The secret directory is located at `./secrets`. You can store any secrets you need in this directory. The server will automatically load the secrets when it starts.

An example for a secret called `my_secret` would look like this:

```bash
# ./secrets/my_secret
my_secret_value
```

This is useful when running on docker so you can copy the secrets into the container without exposing them in the Dockerfile.

## Database selection


### PostgreSQL

We use PostgreSQL (with the pgvector extension) as the database. You will swap the commands you use to generate and run prisma to the following

```bash
poetry run prisma generate --schema postgres/schema.prisma
```

This will generate the Prisma client for PostgreSQL. You will also need to run the PostgreSQL database in a separate container. You can use the `docker-compose.yml` file in the `rnd` directory to run the PostgreSQL database.

```bash
cd autogpt_platform/
docker compose up -d --build
```

You can then run the migrations from the `backend` directory.

```bash
cd ../backend
prisma migrate dev --schema postgres/schema.prisma
```

## Cache and coordination engine

Alongside PostgreSQL and RabbitMQ, the platform depends on a Redis-compatible engine for caching, distributed locking, rate limiting, spend and usage counters, session metadata, pending-message buffers, and the server-sent-event streams that carry agent output to the browser.

Redis is the default. **Valkey is a tested alternative:** it is the engine inside the single-container image, and the Compose stack can be pointed at it with a single variable. Valkey forked from Redis 7.2 and speaks the same protocol — nothing the backend does distinguishes the two.

| | Redis | Valkey |
|---|---|---|
| Docker Compose stack | Default (`redis:7`) | Opt-in, via `REDIS_IMAGE` |
| Single-container image | Not used | The engine it ships |
| Backend CI | Every test leg | One additional, advisory leg |
| Connection settings | `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD` | Identical — nothing to change |

There is no functional reason to prefer one over the other for this workload: both clear the [version floor](#engine-version-floor) below, and the platform uses no Redis modules. Choose on operational grounds — which engine your managed provider offers, and which licence terms you want. The two projects' licences differ and the Redis side has changed more than once, so check the licence of the specific tag you pin.

### Cluster mode is required

The backend always connects with a cluster client, so **a single standalone node will not work** — regardless of which engine you pick. Local development deliberately runs a real multi-shard cluster so that cross-slot bugs surface on a laptop rather than in production.

The self-hosting distributions each bring up a three-shard, no-replica cluster on ports `17000`, `17001` and `17002` (cluster bus ports `27000`–`27002`):

| Distribution | Engine | How the cluster is formed |
|---|---|---|
| Docker Compose stack (`autogpt_platform/docker-compose.platform.yml`) | `redis:7`, or whatever `REDIS_IMAGE` names | Three `redis-server` containers plus a one-shot `redis-init` sidecar that runs `redis-cli --cluster create`. Each shard announces its own Compose hostname. |
| Single-container image (`autogpt_platform/single-container`) | Valkey | Three supervised `valkey-server` processes inside the container, formed by `valkey-cli --cluster create`. Each shard announces `127.0.0.1`. |

{% hint style="info" %}
`make start-core` brings up this cluster along with the platform's other dependencies — PostgreSQL, RabbitMQ, FalkorDB, ClamAV and the database migration job — not a single cache node.
{% endhint %}

FalkorDB is a separate service that also speaks the Redis protocol, but it is the CoPilot graph store and depends on the FalkorDB graph module. It is not part of the cache and coordination layer, and Redis or Valkey cannot serve it.

### Switching the Compose stack to Valkey

`REDIS_IMAGE` sets the image for all three shards and the init sidecar:

```bash
cd autogpt_platform/
REDIS_IMAGE=valkey/valkey:8.1 docker compose up -d deps
```

To make it permanent, set it in `autogpt_platform/.env` — the file `make init-env` creates from `.env.default`, where `REDIS_IMAGE` is listed commented out. Note that this is the file Compose interpolates from; `backend/.env` is passed *into* the containers and cannot reach it.

Nothing else changes. Connection settings are engine-neutral — `REDIS_HOST`, `REDIS_PORT` and `REDIS_PASSWORD` mean the same thing to both engines. (`REDIS_CLUSTER_HOST` and `REDIS_CLUSTER_PORT` take precedence over the first two when they are set.) The service names stay `redis-0`/`redis-1`/`redis-2` with the `redis-init` sidecar, and the Valkey image ships `redis-server` and `redis-cli` as symlinks, so the cluster command lines and health checks in the Compose file work unaltered.

{% hint style="info" %}
The shards run as uid 999 under either engine, pinned in the Compose file. This matters if you write your own override: Valkey's entrypoint only drops privileges when it is invoked as `valkey-server`, and the Compose command lines call the `redis-server` symlink — so an override that sets `image:` without also setting `user: "999:999"` runs the shards as root.
{% endhint %}

{% hint style="warning" %}
`REDIS_IMAGE` arrived with this page. On an earlier checkout the same substitution needs a `docker-compose.override.yml` setting `image:` and `user: "999:999"` on `redis-0`, `redis-1`, `redis-2` and `redis-init`.
{% endhint %}

### Using a managed or external deployment

For a cluster you buy or run yourself — Amazon ElastiCache or Google Memorystore, both of which offer Redis- and Valkey-flavoured clusters, or a self-managed cluster of either engine — the deployment must provide:

- **Cluster mode enabled.** A single-node or cluster-mode-disabled deployment cannot serve this platform, because the backend speaks only the cluster protocol.
- **Sharded pub/sub** (`SPUBLISH`, `SSUBSCRIBE`, `SUNSUBSCRIBE`). Agent output streaming and websocket reconnection depend on it; it is not optional.
- **Announced shard addresses that resolve from the platform**, together with `REDIS_USE_ANNOUNCED_ADDRESS=true`. Without that variable the backend rewrites every shard address to the seed host, keeping only the announced port. Managed clusters give each shard a distinct hostname on a shared port, so the rewrite collapses all shards onto one node and sharded pub/sub is pinned to the wrong shard. The Compose stack already sets this variable; set it yourself if you run the backend outside Compose.

To point the Compose stack at an external cluster, change `REDIS_HOST` and `REDIS_PORT` in the `x-backend-env` block of `docker-compose.platform.yml`. The backend services take their values from there, so editing `backend/.env` alone will not move them. `REDIS_PASSWORD` is the exception: it is absent from that block, so `backend/.env` does set it.

If you would rather not edit a tracked file, set the same variables per backend service in `docker-compose.override.yml` — a service-level `environment:` entry overrides the value merged in from `x-backend-env`. You have to list every backend service you run, which is why the block above is the shorter route.

Then start the dependencies you still need directly instead of through `deps`, which always brings up the bundled shards:

```bash
docker compose up -d db rabbitmq clamav falkordb migrate
```

### Engine version floor

The commands the backend issues imply these minimums, for either engine:

| Requirement | Commands |
|---|---|
| Redis 7.0-equivalent semantics | Sharded pub/sub (`SPUBLISH`/`SSUBSCRIBE`/`SUNSUBSCRIBE`), `EXPIRE … NX` |
| Redis 6.2-equivalent semantics | `LPOP` with a `count` argument, `GETEX` |

Valkey 8.1 and Redis 7 both clear this floor.

The cache and coordination layer uses **no Redis modules** — no `FT.*`, `JSON.*`, `BF.*` or `TS.*` commands appear in the backend — so there is no module bundle to install or license on either engine. It does rely on Redis Streams, server-side Lua (`EVAL`), and transactions within a single hash slot, which a thin protocol proxy may not implement in full.

{% hint style="warning" %}
This guidance covers self-hosting and local development. Behaviour under sustained production load, failover and persistence tuning depends on how you size and operate the deployment, and is outside the scope of these instructions.
{% endhint %}

## AutoGPT Agent Server Advanced set up

This guide walks you through a dockerized set up, with an external DB (postgres)

### Setup

We use the Poetry to manage the dependencies. To set up the project, follow these steps inside this directory:

0. Install Poetry
    ```sh
    pip install poetry
    ```
    
1. Configure Poetry to use .venv in your project directory
    ```sh
    poetry config virtualenvs.in-project true
    ```

2. Enter the poetry shell

   ```sh
   poetry shell
   ```

3. Install dependencies

   ```sh
   poetry install
   ```

4. Copy .env.default to .env

   ```sh
   cp .env.default .env
   ```

5. Generate the Prisma client

   ```sh
   poetry run prisma generate
   ```

   > In case Prisma generates the client for the global Python installation instead of the virtual environment, the current mitigation is to just uninstall the global Prisma package:
   >
   > ```sh
   > pip uninstall prisma
   > ```
   >
   > Then run the generation again. The path _should_ look something like this:  
   > `<some path>/pypoetry/virtualenvs/backend-TQIRSwR6-py3.12/bin/prisma`

6. Run the postgres database from the /rnd folder

   ```sh
   cd autogpt_platform/
   docker compose up -d
   ```

7. Run the migrations (from the backend folder)

   ```sh
   cd ../backend
   prisma migrate deploy
   ```

### Running The Server

#### Starting the server directly

Run the following command:

```sh
poetry run app
```
