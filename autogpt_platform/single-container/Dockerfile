# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

# Build the relocatable frontend on glibc so its standalone native modules can
# be copied into the Debian appliance image. Browser-facing service URLs are
# intentionally same-origin paths; the in-container reverse proxy decides
# where they land at runtime.
FROM node:24.18.0-bookworm-slim@sha256:6f7b03f7c2c8e2e784dcf9295400527b9b1270fd37b7e9a7285cf83b6951452d AS frontend

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

FROM node:24.18.0-bookworm-slim@sha256:6f7b03f7c2c8e2e784dcf9295400527b9b1270fd37b7e9a7285cf83b6951452d AS node-runtime

ARG NPM_VERSION=12.0.2
RUN npm install --global "npm@${NPM_VERSION}" \
    && npm cache clean --force

# RabbitMQ's official image ships a self-contained Erlang/OpenSSL runtime.
# Pin the current supported 4.1.x patch by its multi-platform manifest.
FROM rabbitmq:4.1.8@sha256:b921426eabfb39e201c60f0623f6b45a3160cf97b2dc3af2f5340365f65b5be3 AS rabbitmq

# FalkorDB provides the Graphiti memory store. Copy only the server and module;
# its browser UI is not part of the appliance image.
FROM falkordb/falkordb-server:v4.20.1@sha256:8b676c2e72271f5d06541136eb214b0db50180636cc86f7ee38d63287e9b8d13 AS falkordb

# Seed ClamAV's persistent signature volume so a cold start can scan before the
# first network refresh completes. Runtime binaries come from Debian packages.
FROM clamav/clamav-debian:1.5.1@sha256:c102cd8fffe36fb27808a714f2d9c977e28549b3aae1ea0de7d7ac7814f8ded7 AS clamav-seed

# autogpt-backend is supplied by Docker Bake from the existing backend
# Dockerfile's server target. This keeps the backend image contract independent
# from the opt-in single-container distribution.
FROM autogpt-backend AS single-container

ARG IMAGE_VERSION=dev
ARG VCS_REF=unknown

# The normal backend image stores generated Prisma engines under root's cache.
# Appliance services run unprivileged, so relocate those inherited engines to a
# traversable path without changing the backend image used by cloud deployments.
RUN install -d -m 0755 /opt/prisma-python \
    && cp -a /root/.cache/prisma-python/binaries /opt/prisma-python/binaries \
    && set -- /opt/prisma-python/binaries/*/*/node_modules/prisma/build/index.js \
    && test "$#" -eq 1 \
    && prisma_cli="$1" \
    && test -f "${prisma_cli}" \
    && prisma_node_modules="${prisma_cli%/prisma/build/index.js}" \
    && set -- "${prisma_node_modules}"/prisma/query-engine-* \
    && test "$#" -eq 1 \
    && query_engine="$1" \
    && test -f "${query_engine}" \
    && test -x "${query_engine}" \
    && ln -s "${prisma_node_modules}" /opt/prisma-python/binaries/node_modules \
    && ln -s "${query_engine}" /opt/prisma-python/query-engine

LABEL org.opencontainers.image.title="AutoGPT Platform single-container" \
      org.opencontainers.image.description="Experimental single-node AutoGPT Platform distribution" \
      org.opencontainers.image.source="https://github.com/Significant-Gravitas/AutoGPT" \
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
        postgresql-15 \
        postgresql-client-15 \
        postgresql-15-pgvector \
        procps \
        supervisor \
        tini \
        valkey-server \
        valkey-tools \
    && rm -f /usr/sbin/policy-rc.d \
        /etc/ssl/private/ssl-cert-snakeoil.key \
        /etc/ssl/certs/ssl-cert-snakeoil.pem \
    && find /etc/ssl/certs -maxdepth 1 -type l \
        -lname '*ssl-cert-snakeoil.pem' -delete \
    && test ! -e /etc/ssl/private/ssl-cert-snakeoil.key \
    && test ! -e /etc/ssl/certs/ssl-cert-snakeoil.pem \
    && test -z "$(find /etc/ssl/certs -maxdepth 1 -type l \
        -lname '*ssl-cert-snakeoil.pem' -print -quit)" \
    && rm -rf /var/lib/apt/lists/* /var/lib/postgresql/15/main \
    && find /var/lib/clamav -mindepth 1 -maxdepth 1 -delete

# Node 24 is required by the frontend. Backend Prisma and agent-browser also
# use this binary through PATH; their npm packages remain from the server stage.
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=frontend /usr/local/LICENSE /usr/share/licenses/nodejs/LICENSE
RUN rm -rf /usr/lib/node_modules/npm
COPY --from=node-runtime /usr/local/lib/node_modules/npm /usr/lib/node_modules/npm

# RabbitMQ + Erlang runtime and their bundled third-party license files.
COPY --from=rabbitmq /opt/erlang /opt/erlang
COPY --from=rabbitmq /opt/openssl /opt/openssl
COPY --from=rabbitmq /opt/rabbitmq /opt/rabbitmq

# FalkorDB server only (no bundled Browser UI).
COPY --from=falkordb /usr/local/bin/redis-server /opt/falkordb/redis-server
COPY --from=falkordb /var/lib/falkordb/bin/falkordb.so /opt/falkordb/falkordb.so

# Cold-start antivirus definitions. entrypoint copies these into /data once.
COPY --from=clamav-seed --chown=102:104 /var/lib/clamav /opt/clamav-seed

# Next standalone server and immutable assets.
COPY --from=frontend /app/.next/standalone /app/frontend
COPY --from=frontend /app/.next/static /app/frontend/.next/static
COPY --from=frontend /app/public /app/frontend/public

# Next's standalone trace over-includes Orval's build-only esbuild dependency.
# It is never used by the production server and must not ship as runtime code.
RUN rm -rf \
        /app/frontend/node_modules/.pnpm/esbuild@* \
        /app/frontend/node_modules/.pnpm/@esbuild+linux-*@* \
    && find /app/frontend/node_modules -type l \
        \( -name esbuild -o -path '*/node_modules/@esbuild/linux-*' \) -delete

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
     autogpt_platform/single-container/run-frontend.sh \
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
    && site_packages="$(/app/autogpt_platform/backend/.venv/bin/python -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && rm -f \
        "${site_packages}/ldclient/testing/selfsigned.key" \
        "${site_packages}/ldclient/testing/selfsigned.pem" \
    && test ! -e "${site_packages}/ldclient/testing/selfsigned.key" \
    && test ! -e "${site_packages}/ldclient/testing/selfsigned.pem"

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
    PRISMA_QUERY_ENGINE_BINARY=/opt/prisma-python/query-engine \
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
