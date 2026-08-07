# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# ============================ DEPENDENCY BUILDER ============================ #

FROM debian:13-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN echo 'Acquire::http::Pipeline-Depth 0;\nAcquire::http::No-Cache true;\nAcquire::BrokenProxy true;\n' > /etc/apt/apt.conf.d/99fixbadproxy

# Install Node.js repository key and setup
RUN apt-get update --allow-releaseinfo-change --fix-missing \
    && apt-get install -y curl ca-certificates gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list

# Update package list and install Python, Node.js, and build dependencies
RUN apt-get update \
    && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3-pip \
    build-essential \
    libpq5 \
    libz-dev \
    libssl-dev \
    postgresql-client \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME=/opt/poetry
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH=/opt/poetry/bin:$PATH

RUN pip3 install poetry --break-system-packages

# Copy and install dependencies
COPY autogpt_platform/autogpt_libs /app/autogpt_platform/autogpt_libs
COPY autogpt_platform/backend/poetry.lock autogpt_platform/backend/pyproject.toml /app/autogpt_platform/backend/
WORKDIR /app/autogpt_platform/backend
RUN poetry install --no-ansi --no-root

# Generate Prisma client
COPY autogpt_platform/backend/schema.prisma ./
COPY autogpt_platform/backend/backend/data/partial_types.py ./backend/data/partial_types.py
COPY autogpt_platform/backend/scripts/gen_prisma_types_stub.py ./scripts/
RUN poetry run prisma generate && poetry run gen-prisma-stub

# =============================== DB MIGRATOR =============================== #

# Lightweight migrate stage - only needs Prisma CLI, not full Python environment
FROM debian:13-slim AS migrate

WORKDIR /app/autogpt_platform/backend

ENV DEBIAN_FRONTEND=noninteractive

# Install only what's needed for prisma migrate: Node.js and minimal Python for prisma-python
# libatomic1: debian:13-slim no longer ships it, but the Prisma CLI's bundled
# Node (copied from the builder, which had it via build-essential) dynamically
# links libatomic.so.1 — without it `node` exits 127 and migrate crash-loops.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3-pip \
    ca-certificates \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Node.js from builder (needed for Prisma CLI)
COPY --from=builder /usr/bin/node /usr/bin/node
COPY --from=builder /usr/lib/node_modules /usr/lib/node_modules
COPY --from=builder /usr/bin/npm /usr/bin/npm

# Copy Prisma binaries
COPY --from=builder /root/.cache/prisma-python/binaries /root/.cache/prisma-python/binaries

# Install prisma-client-py directly (much smaller than copying full venv)
RUN pip3 install prisma>=0.15.0 --break-system-packages

COPY autogpt_platform/backend/schema.prisma ./
COPY autogpt_platform/backend/backend/data/partial_types.py ./backend/data/partial_types.py
COPY autogpt_platform/backend/scripts/gen_prisma_types_stub.py ./scripts/
COPY autogpt_platform/backend/migrations ./migrations

# ============================== BACKEND SERVER ============================== #

FROM debian:13-slim AS docs-filter
# The copilot docs tools read markdown only — strip everything else
# (images, scripts, configs) so no unused or untrusted files ship in the
# server image or its layer history.
COPY docs /docs
RUN find /docs -type f ! -name '*.md' ! -name '*.mdx' -delete \
    && find /docs -type d -empty -delete

FROM debian:13-slim AS server

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, FFmpeg, ImageMagick, and CLI tools for agent use.
# bubblewrap provides OS-level sandbox (whitelist-only FS + no network)
# for the bash_exec MCP tool (fallback when E2B is not configured).
# Using --no-install-recommends saves ~650MB by skipping unnecessary deps like llvm, mesa, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3-pip \
    ffmpeg \
    imagemagick \
    jq \
    ripgrep \
    tree \
    bubblewrap \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry (build-time only, for `poetry install --only-root` to create entry points)
COPY --from=builder /usr/local/lib/python3* /usr/local/lib/python3*
COPY --from=builder /usr/local/bin/poetry /usr/local/bin/poetry
# Copy Node.js installation for Prisma and agent-browser.
# npm/npx are symlinks in the builder (-> ../lib/node_modules/npm/bin/*-cli.js);
# COPY resolves them to regular files, breaking require() paths.  Recreate as
# proper symlinks so npm/npx can find their modules.
COPY --from=builder /usr/bin/node /usr/bin/node
COPY --from=builder /usr/lib/node_modules /usr/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/bin/npx
COPY --from=builder /root/.cache/prisma-python/binaries /root/.cache/prisma-python/binaries

# Install agent-browser (Copilot browser tool) using the system chromium package.
# Chrome for Testing (the binary agent-browser downloads via `agent-browser install`)
# has no ARM64 builds, so we use the distro-packaged chromium instead — verified to
# work with agent-browser via Docker tests on arm64; amd64 is validated in CI.
# Note: system chromium tracks the Debian package schedule rather than a pinned
# Chrome for Testing release. If agent-browser requires a specific Chrome version,
# verify compatibility against the chromium package version in the base image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g agent-browser \
    && rm -rf /tmp/* /root/.npm

ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium

WORKDIR /app/autogpt_platform/backend

# Copy only the .venv from builder (not the entire /app directory)
# The .venv includes the generated Prisma client
COPY --from=builder /app/autogpt_platform/backend/.venv ./.venv
ENV PATH="/app/autogpt_platform/backend/.venv/bin:$PATH"

# Copy dependency files + autogpt_libs (path dependency)
COPY autogpt_platform/autogpt_libs /app/autogpt_platform/autogpt_libs
COPY autogpt_platform/backend/poetry.lock autogpt_platform/backend/pyproject.toml ./

# Copy backend code + docs (markdown only — see docs-filter stage)
COPY autogpt_platform/backend ./
COPY --from=docs-filter /docs /app/docs
# Install the project package to create entry point scripts in .venv/bin/
# (e.g., rest, executor, ws, db, scheduler, notification - see [tool.poetry.scripts])
RUN POETRY_VIRTUALENVS_CREATE=true POETRY_VIRTUALENVS_IN_PROJECT=true \
    poetry install --no-ansi --only-root

ENV PORT=8000

CMD ["rest"]

# ======================= SINGLE-CONTAINER FRONTEND ======================== #

# Build the relocatable frontend on glibc so its standalone native modules can
# be copied into the Debian appliance image. Browser-facing service URLs are
# intentionally same-origin paths; the in-container reverse proxy decides
# where they land at runtime.
FROM node:24.18.0-bookworm-slim@sha256:6f7b03f7c2c8e2e784dcf9295400527b9b1270fd37b7e9a7285cf83b6951452d AS single-container-frontend

WORKDIR /app
RUN corepack enable

COPY autogpt_platform/frontend/package.json autogpt_platform/frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm \
    pnpm install --frozen-lockfile

COPY autogpt_platform/frontend/ ./

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_AGPT_SERVER_URL=/_agpt/api \
    NEXT_PUBLIC_AGPT_WS_SERVER_URL=/_agpt/ws \
    NEXT_PUBLIC_FRONTEND_BASE_URL="" \
    NEXT_PUBLIC_APP_ENV=local \
    NEXT_PUBLIC_BEHAVE_AS=LOCAL \
    NEXT_PUBLIC_LAUNCHDARKLY_ENABLED=false \
    NEXT_PUBLIC_SOURCEMAPS=false \
    NEXT_PUBLIC_TURNSTILE=disabled \
    NEXT_PUBLIC_VAPID_PUBLIC_KEY="" \
    BETTER_AUTH_SECRET=build-only-placeholder-not-used-at-runtime \
    DATABASE_URL=postgresql://postgres@127.0.0.1:5432/postgres

RUN pnpm run generate:api \
    && NODE_OPTIONS="--max-old-space-size=8192" pnpm build \
    && rm -f .env .env.local .env.production

# ====================== SINGLE-CONTAINER DEPENDENCIES ===================== #

# RabbitMQ's official image ships a self-contained Erlang/OpenSSL runtime.
# Pin the current supported 4.1.x patch by its multi-platform manifest.
FROM rabbitmq:4.1.8@sha256:b921426eabfb39e201c60f0623f6b45a3160cf97b2dc3af2f5340365f65b5be3 AS single-container-rabbitmq

# FalkorDB provides the Graphiti memory store. Copy only the server and module;
# its browser UI is not part of the appliance image.
FROM falkordb/falkordb-server:v4.20.1@sha256:8b676c2e72271f5d06541136eb214b0db50180636cc86f7ee38d63287e9b8d13 AS single-container-falkordb

# Seed ClamAV's persistent signature volume so a cold start can scan before the
# first network refresh completes. Runtime binaries come from Debian packages.
FROM clamav/clamav-debian:1.5.1@sha256:c102cd8fffe36fb27808a714f2d9c977e28549b3aae1ea0de7d7ac7814f8ded7 AS single-container-clamav-seed

# ======================== SINGLE-CONTAINER IMAGE ========================== #

FROM server AS single-container

ARG POSTGRES_MAJOR=15
ARG IMAGE_VERSION=dev
ARG VCS_REF=unknown

# The normal backend image stores generated Prisma engines under root's cache.
# Appliance services run unprivileged, so copy those engines to a traversable
# path without changing the existing server target used by cloud deployments.
COPY --from=builder /root/.cache/prisma-python/binaries /opt/prisma-python/binaries

LABEL org.opencontainers.image.title="AutoGPT Platform single-container" \
      org.opencontainers.image.description="Experimental single-node AutoGPT Platform distribution" \
      org.opencontainers.image.source="https://github.com/Significant-Gravitas/AutoGPT" \
      org.opencontainers.image.documentation="https://github.com/Significant-Gravitas/AutoGPT/blob/dev/docs/platform/single-container.md" \
      org.opencontainers.image.licenses="LicenseRef-PolyForm-Shield-1.0.0 AND SSPL-1.0" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

ENV DEBIAN_FRONTEND=noninteractive

# Block service auto-start during package installation; Supervisor owns every
# daemon at runtime. PostgreSQL comes from PGDG to retain PG15 parity on Debian
# 13, while Valkey comes from Debian's security-maintained BSD build.
# Persistent volume ownership must remain stable across package upgrades.
# Pre-create the accounts with the IDs used by the first image release; fail
# the build rather than silently shifting them if the base ever claims an ID.
RUN printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d \
    && chmod 0755 /usr/sbin/policy-rc.d \
    && groupadd --system --gid 102 postgres \
    && useradd --system --uid 100 --gid postgres --home-dir /var/lib/postgresql --shell /bin/bash postgres \
    && groupadd --system --gid 104 clamav \
    && useradd --system --uid 102 --gid clamav --home-dir /var/lib/clamav --shell /bin/false clamav \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        postgresql-common \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
        | gpg --dearmor -o /etc/apt/keyrings/postgresql.gpg \
    && printf 'deb [signed-by=/etc/apt/keyrings/postgresql.gpg] https://apt.postgresql.org/pub/repos/apt trixie-pgdg main\n' \
        > /etc/apt/sources.list.d/pgdg.list \
    && printf 'create_main_cluster = false\n' > /etc/postgresql-common/createcluster.conf \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        clamav \
        clamav-daemon \
        clamav-freshclam \
        clamdscan \
        libgomp1 \
        netcat-openbsd \
        nginx \
        openssl \
        postgresql-${POSTGRES_MAJOR} \
        postgresql-client-${POSTGRES_MAJOR} \
        postgresql-${POSTGRES_MAJOR}-pgvector \
        procps \
        supervisor \
        tini \
        valkey-server \
        valkey-tools \
    && rm -f /usr/sbin/policy-rc.d \
    && rm -rf /var/lib/apt/lists/* /var/lib/postgresql/${POSTGRES_MAJOR}/main \
    && find /var/lib/clamav -mindepth 1 -maxdepth 1 -delete

# Node 24 is required by the frontend. Backend Prisma and agent-browser also
# use this binary through PATH; their npm packages remain from the server stage.
COPY --from=single-container-frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=single-container-frontend /usr/local/LICENSE /usr/share/licenses/nodejs/LICENSE

# RabbitMQ + Erlang runtime and their bundled third-party license files.
COPY --from=single-container-rabbitmq /opt/erlang /opt/erlang
COPY --from=single-container-rabbitmq /opt/openssl /opt/openssl
COPY --from=single-container-rabbitmq /opt/rabbitmq /opt/rabbitmq

# FalkorDB server only (no bundled Browser UI).
COPY --from=single-container-falkordb /usr/local/bin/redis-server /opt/falkordb/redis-server
COPY --from=single-container-falkordb /var/lib/falkordb/bin/falkordb.so /opt/falkordb/falkordb.so

# Cold-start antivirus definitions. entrypoint copies these into /data once.
COPY --from=single-container-clamav-seed --chown=102:104 /var/lib/clamav /opt/clamav-seed

# Next standalone server and immutable assets.
COPY --from=single-container-frontend /app/.next/standalone /app/frontend
COPY --from=single-container-frontend /app/.next/static /app/frontend/.next/static
COPY --from=single-container-frontend /app/public /app/frontend/public

# Runtime orchestration, proxy, health tooling, and database compatibility shim.
COPY autogpt_platform/single-container/bootstrap.sh \
     autogpt_platform/single-container/common.sh \
     autogpt_platform/single-container/disabled-service.sh \
     autogpt_platform/single-container/entrypoint.sh \
     autogpt_platform/single-container/fatal_listener.py \
     autogpt_platform/single-container/healthcheck.sh \
     autogpt_platform/single-container/probe.py \
     autogpt_platform/single-container/promote-admin.sh \
     autogpt_platform/single-container/run-app.sh \
     autogpt_platform/single-container/run-optional-app.sh \
     autogpt_platform/single-container/run-service.sh \
     autogpt_platform/single-container/runtime_config.py \
     autogpt_platform/single-container/watchdog.sh \
     /opt/autogpt/single-container/
COPY autogpt_platform/single-container/clamav /opt/autogpt/single-container/clamav
COPY autogpt_platform/single-container/nginx /opt/autogpt/single-container/nginx
COPY autogpt_platform/single-container/python /opt/autogpt/single-container/python
COPY autogpt_platform/single-container/rabbitmq /opt/autogpt/single-container/rabbitmq
COPY autogpt_platform/single-container/supervisor /opt/autogpt/single-container/supervisor
COPY autogpt_platform/db/init/00-init.sql /opt/autogpt/single-container/00-init.sql
COPY autogpt_platform/LICENSE.md /usr/share/licenses/autogpt-platform/LICENSE.md
COPY autogpt_platform/single-container/licenses/FalkorDB-SSPL-1.0.txt /usr/share/licenses/falkordb/SSPL-1.0.txt
COPY autogpt_platform/single-container/licenses/FalkorDB-SOURCE.txt /usr/share/licenses/falkordb/SOURCE.txt
COPY autogpt_platform/single-container/licenses/Redis-LICENSE.txt /usr/share/licenses/redis/LICENSE.txt
COPY autogpt_platform/single-container/licenses/Redis-REDISCONTRIBUTIONS.txt /usr/share/licenses/redis/REDISCONTRIBUTIONS.txt
COPY autogpt_platform/single-container/licenses/Redis-SOURCE.txt /usr/share/licenses/redis/SOURCE.txt
COPY autogpt_platform/single-container/licenses/redis/deps /usr/share/licenses/redis/deps
COPY autogpt_platform/single-container/licenses/redis/src /usr/share/licenses/redis/src

RUN groupadd --system --gid 10001 autogpt \
    && useradd --system --uid 10001 --gid autogpt --home-dir /data/home --shell /usr/sbin/nologin autogpt \
    && groupadd --system --gid 10002 rabbitmq \
    && useradd --system --uid 10002 --gid rabbitmq --home-dir /data/rabbitmq --shell /usr/sbin/nologin rabbitmq \
    && groupadd --system --gid 10003 autogpt-valkey \
    && useradd --system --uid 10003 --gid autogpt-valkey --home-dir /data/valkey --shell /usr/sbin/nologin autogpt-valkey \
    && groupadd --system --gid 10004 autogpt-falkor \
    && useradd --system --uid 10004 --gid autogpt-falkor --home-dir /data/falkordb --shell /usr/sbin/nologin autogpt-falkor \
    && chmod -R a+rX /opt/autogpt/single-container /usr/share/licenses/autogpt-platform /usr/share/licenses/falkordb /usr/share/licenses/redis \
    && find /opt/autogpt/single-container -type f \( -name '*.sh' -o -name 'entrypoint' -o -name 'autogpt-*' \) -exec chmod 0755 {} + \
    && ln -sf /opt/autogpt/single-container/healthcheck.sh /usr/local/bin/autogpt-healthcheck \
    && ln -sf /opt/autogpt/single-container/promote-admin.sh /usr/local/bin/autogpt-admin \
    && rm -f /app/autogpt_platform/backend/config.json \
    && ln -s /data/config/backend.json /app/autogpt_platform/backend/config.json \
    && rm -rf /app/frontend/.next/cache \
    && ln -s /data/cache/next /app/frontend/.next/cache \
    && rm -f \
        /app/autogpt_platform/backend/.venv/lib/python3.13/site-packages/ldclient/testing/selfsigned.key \
        /app/autogpt_platform/backend/.venv/lib/python3.13/site-packages/ldclient/testing/selfsigned.pem \
    && test -z "$(find /app/frontend /app/autogpt_platform/backend \( ! -user root -o ! -group root \) -print -quit)" \
    && test -z "$(find /app/frontend /app/autogpt_platform/backend \( -type f -o -type d \) -perm /022 -print -quit)"

ENV PATH="/opt/rabbitmq/sbin:/opt/erlang/bin:/opt/openssl/bin:/usr/lib/postgresql/15/bin:/app/autogpt_platform/backend/.venv/bin:${PATH}" \
    ERLANG_INSTALL_PATH_PREFIX=/opt/erlang \
    OPENSSL_INSTALL_PATH_PREFIX=/opt/openssl \
    RABBITMQ_HOME=/opt/rabbitmq \
    RABBITMQ_DATA_DIR=/data/rabbitmq \
    RUNNING_UNDER_SYSTEMD=true \
    PGDATA=/data/postgres \
    PORT=3001 \
    HOSTNAME=127.0.0.1 \
    LANG=C.UTF-8 \
    LANGUAGE=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PRISMA_BINARY_CACHE_DIR=/opt/prisma-python/binaries \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORCE_FLAG_GRAPHITI_MEMORY=true \
    AUTOGPT_PUBLIC_URL=http://localhost:3000

WORKDIR /app/autogpt_platform/backend

VOLUME ["/data"]
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=45s --start-period=5m --retries=10 \
    CMD ["/usr/local/bin/autogpt-healthcheck"]

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/autogpt/single-container/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-n", "-c", "/opt/autogpt/single-container/supervisor/supervisord.conf"]

# Existing infrastructure builds this Dockerfile without --target. Preserve the
# historical default output even though the opt-in appliance target is declared
# after server.
FROM server AS server-default
