# db-migrate

Rust-based database migration tool for AutoGPT Platform - migrates from Supabase to GCP Cloud SQL.

## Features

- Stream data efficiently using PostgreSQL COPY protocol
- Verify both databases match with row counts and checksums
- Migrate auth data (passwords, OAuth IDs) from Supabase auth.users
- Check all triggers and functions are in place
- Progress bars and detailed logging

## Build

```bash
cd backend/tools/db-migrate
cargo build --release
```

The binary will be at `target/release/db-migrate`.

## Usage

```bash
# Set environment variables
export SOURCE_URL="postgresql://postgres:password@db.xxx.supabase.co:5432/postgres?schema=platform"
export DEST_URL="postgresql://postgres:password@ipaddress:5432/postgres?schema=platform"

# Or pass as arguments
db-migrate --source "..." --dest "..." <command>
```

## Commands

### Quick Migration (Users + Auth only)

For testing login/signup ASAP:

```bash
db-migrate quick
```

Migrates: User, Profile, UserOnboarding, UserBalance + auth data

### Full Migration

```bash
db-migrate full
```

Migrates all tables (excluding large execution history by default).

### Schema Only

```bash
db-migrate schema
```

### Data Only

```bash
# All tables (excluding large)
db-migrate data

# Specific table
db-migrate data --table User
```

### Auth Only

```bash
db-migrate auth
```

### Verify

```bash
# Row counts
db-migrate verify

# Include functions and triggers
db-migrate verify --check-functions
```

### Table Sizes

```bash
db-migrate table-sizes
```

### Stream Large Tables

After initial migration, stream execution history:

```bash
# All large tables
db-migrate stream-large

# Specific table
db-migrate stream-large --table AgentGraphExecution
```

## Docker / Kubernetes

Build and run in a container:

```dockerfile
FROM rust:1.75-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libpq5 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/db-migrate /usr/local/bin/
ENTRYPOINT ["db-migrate"]
```

## Large Tables (Excluded by Default)

These tables contain execution history (~37GB) and are excluded from initial migration:

- AgentGraphExecution (1.3 GB)
- AgentNodeExecution (6 GB)
- AgentNodeExecutionInputOutput (30 GB)
- AgentNodeExecutionKeyValueData
- NotificationEvent (94 MB)

Use `stream-large` command to migrate these after the initial migration.
