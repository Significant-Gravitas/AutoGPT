# syntax=docker/dockerfile:1
#
# AutoGPT Platform (https://github.com/Significant-Gravitas/AutoGPT), built for Anthropic's OSS Scanner. The scanner
# builds `dev` with this file, then scans the result offline, reading threat_model.md (its enrolment is
# projects/autogpt in https://github.com/anthropics/oss-scanner). A change to how the platform builds or runs its
# tests should update this file in the same pull request.
#
# This builds autogpt_platform/ the way its maintainers and CI do:
#   - the FastAPI backend and its services (REST, WebSocket, executor, scheduler, AutoPilot) and the shared
#     autogpt_libs package: Python 3.13, Poetry and the generated Prisma client
#   - the Next.js frontend, which also hosts the Better Auth service: Node, pnpm and a production build
# classic/ is unsupported legacy code and is not installed (see threat_model.md).
#
# CI runs the backend tests against PostgreSQL 15 with pgvector, a 3-shard Redis cluster, RabbitMQ and ClamAV,
# each in a container of its own. This image has them all installed instead, with a database already migrated to
# the checkout:
#
#   autogpt-services start            PostgreSQL :5432, Redis cluster :17000-17002, RabbitMQ :5672, ClamAV :3310
#   . /etc/autogpt-scanner/test.env   the environment CI runs the backend tests in
#   cd /src/autogpt_platform/backend && poetry run pytest -q backend/api/features/library
#
# threat_model.md also says how to run the whole platform offline.

# RabbitMQ with the Erlang and OpenSSL it is built against, at the version backend CI tests with.
FROM rabbitmq:4.1.4 AS rabbitmq

FROM debian:trixie

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    COREPACK_HOME=/opt/corepack \
    COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    NEXT_TELEMETRY_DISABLED=1 \
    PGDATA=/var/lib/postgresql/data \
    ERLANG_INSTALL_PATH_PREFIX=/opt/erlang \
    OPENSSL_INSTALL_PATH_PREFIX=/opt/openssl \
    RABBITMQ_HOME=/opt/rabbitmq \
    RABBITMQ_NODENAME=rabbit@localhost \
    RABBITMQ_CONFIG_FILE=/etc/rabbitmq/rabbitmq.conf \
    RABBITMQ_MNESIA_BASE=/var/lib/autogpt-services/rabbitmq \
    RABBITMQ_LOG_BASE=/var/log/autogpt-services \
    ERL_EPMD_ADDRESS=127.0.0.1 \
    PATH=/opt/poetry/bin:/usr/lib/postgresql/15/bin:/opt/rabbitmq/sbin:/opt/erlang/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Python 3.13 (what the production image runs) and what building wheels needs; PostgreSQL 15 and pgvector from
# PGDG, the versions CI and production use; Valkey for the Redis cluster; ClamAV, which scans every upload; and
# the programs the backend's blocks and AutoPilot tools call (ffmpeg, ImageMagick, bubblewrap for the sandboxed
# shell, ...).
RUN printf '#!/bin/sh\nexit 101\n' > /usr/sbin/policy-rc.d \
    && chmod 0755 /usr/sbin/policy-rc.d \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg postgresql-common \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /etc/apt/keyrings/postgresql.gpg \
    && echo 'deb [signed-by=/etc/apt/keyrings/postgresql.gpg] https://apt.postgresql.org/pub/repos/apt trixie-pgdg main' \
        > /etc/apt/sources.list.d/pgdg.list \
    && echo 'create_main_cluster = false' > /etc/postgresql-common/createcluster.conf \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git pkg-config openssl procps psmisc netcat-openbsd less \
        python3.13 python3.13-dev python3.13-venv \
        libpq-dev libffi-dev libssl-dev zlib1g-dev \
        postgresql-15 postgresql-client-15 postgresql-15-pgvector \
        valkey-server valkey-tools \
        clamav-daemon clamdscan \
        ffmpeg imagemagick jq ripgrep tree bubblewrap libatomic1 \
    && rm -f /usr/sbin/policy-rc.d \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rabbitmq /opt/erlang /opt/erlang
COPY --from=rabbitmq /opt/openssl /opt/openssl
COPY --from=rabbitmq /opt/rabbitmq /opt/rabbitmq

# The expert roster and marketplace skills live in a separate repository. Backend CI checks it out and points
# SKILLS_CATALOG_PATH at it (test.env does the same), so no test needs the GitHub API.
RUN git clone --depth 1 https://github.com/Significant-Gravitas/skills-catalog /opt/skills-catalog

# scanner contract: the checkout lives inside the image, at /src. Dockerfile.dockerignore, next to this file, stops
# the repository's .dockerignore (written for the product images) from leaving parts of the checkout out.
COPY . /src

# Node at the major version in .nvmrc, and pnpm through corepack.
RUN node_major="$(tr -dc '0-9.' < /src/.nvmrc | cut -d. -f1)" \
    && test -n "${node_major}" \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${node_major}.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global corepack \
    && corepack enable \
    && node --version

# The backend: Poetry at the version that wrote poetry.lock (CI does the same) in a venv of its own, then the
# backend and its dev dependencies (pytest and friends) into autogpt_platform/backend/.venv, and the Prisma client.
WORKDIR /src/autogpt_platform/backend
RUN poetry_version="$(sed -n '1s/.*by Poetry \([0-9][0-9.]*\).*/\1/p' poetry.lock)" \
    && test -n "${poetry_version}" \
    && python3.13 -m venv /opt/poetry/venv \
    && /opt/poetry/venv/bin/pip install --no-cache-dir "poetry==${poetry_version}" \
    && install -d /opt/poetry/bin \
    && ln -s ../venv/bin/poetry /opt/poetry/bin/poetry \
    && poetry env use python3.13 \
    && poetry install --no-ansi \
    && poetry run prisma generate \
    && poetry run gen-prisma-stub \
    && rm -rf /root/.cache/pypoetry \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# tiktoken downloads its encodings the first time they are used, and the backend uses one as soon as
# backend/api/features/search/embeddings.py is imported, which most test modules do. Fetch them all now.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN poetry run python - <<'EOF'
import tiktoken

for name in tiktoken.list_encoding_names():
    try:
        tiktoken.get_encoding(name)
    except Exception as error:
        if name in ("cl100k_base", "o200k_base"):
            raise
        print(f"WARNING: tiktoken encoding {name} is not cached: {error}")
EOF

# The frontend: its pinned pnpm, locked dependencies, the generated API client and a production build. Its .env
# (gitignored) points the Better Auth service at the local database. pnpm's store (node_modules holds copies of it)
# and Next's build cache take about 5.5 GB that nothing reads once the build is done.
WORKDIR /src/autogpt_platform/frontend
RUN corepack install \
    && pnpm install --frozen-lockfile \
    && pnpm run generate:api \
    && sed -e 's|^DATABASE_URL=.*|DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres|' \
           -e 's|^AUTH_DB_SCHEMA=.*|AUTH_DB_SCHEMA=public|' \
           -e "s|^BETTER_AUTH_SECRET=.*|BETTER_AUTH_SECRET=$(openssl rand -hex 32)|" \
           .env.default > .env \
    && (NEXT_SKIP_BUILD_CHECKS=true timeout 1500 pnpm build \
        || echo 'WARNING: next build failed; `pnpm dev` still serves the frontend') \
    && rm -rf "$(pnpm store path)" .next/cache

COPY --chmod=0755 <<'EOF' /usr/local/bin/autogpt-services
#!/usr/bin/env bash
# autogpt-services start|stop|restart|status [postgres] [redis] [rabbitmq] [clamav]
#
# The services AutoGPT's backend and its tests use, all on localhost, so they work without a network:
#   postgres   PostgreSQL 15 + pgvector on :5432, user postgres (trust auth, any password)
#   redis      a 3-shard Redis cluster (Valkey) on :17000-17002, as in CI and production
#   rabbitmq   RabbitMQ on :5672, user rabbitmq_user_default (password in /etc/autogpt-scanner/secrets.env)
#   clamav     ClamAV on :3310; its only signature is the EICAR test file's
# Logs are in /var/log/autogpt-services.
set -euo pipefail

# A login shell's /etc/profile resets PATH, so don't rely on it.
export PATH="/usr/lib/postgresql/15/bin:/opt/rabbitmq/sbin:/opt/erlang/bin:$PATH"
. /etc/autogpt-scanner/secrets.env

LOGS=/var/log/autogpt-services
STATE=/var/lib/autogpt-services
PG_BIN=/usr/lib/postgresql/15/bin
REDIS_PORTS=(17000 17001 17002)

mkdir -p "$LOGS" "$STATE"

# wait_for SECONDS WHAT COMMAND...: retry COMMAND once a second until it succeeds.
wait_for() {
  local seconds=$1 what=$2
  shift 2
  for _ in $(seq "$seconds"); do
    if "$@" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "autogpt-services: $what is not up after ${seconds}s; see $LOGS" >&2
  return 1
}

postgres_up() { "$PG_BIN/pg_isready" -h localhost -p 5432 -q; }
postgres_start() {
  install -d -o postgres -g postgres -m 2775 /run/postgresql
  touch "$LOGS/postgres.log" && chown postgres:postgres "$LOGS/postgres.log"
  postgres_up && return 0
  runuser -u postgres -- "$PG_BIN/pg_ctl" -D "$PGDATA" -l "$LOGS/postgres.log" -w -t 60 start >/dev/null
  wait_for 30 PostgreSQL postgres_up
}
postgres_stop() {
  runuser -u postgres -- "$PG_BIN/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null 2>&1 || true
}

redis_up() {
  local port
  for port in "${REDIS_PORTS[@]}"; do
    valkey-cli -p "$port" cluster info 2>/dev/null | grep -q '^cluster_state:ok' || return 1
  done
}
redis_start() {
  redis_up && return 0
  redis_stop
  local port
  for port in "${REDIS_PORTS[@]}"; do
    rm -rf "${STATE:?}/valkey/$port" && mkdir -p "$STATE/valkey/$port"
    valkey-server --port "$port" --bind 127.0.0.1 --daemonize yes \
      --dir "$STATE/valkey/$port" --logfile "$LOGS/valkey-$port.log" \
      --cluster-enabled yes --cluster-config-file nodes.conf --cluster-node-timeout 5000 \
      --cluster-require-full-coverage no --save '' --appendonly no
  done
  for port in "${REDIS_PORTS[@]}"; do wait_for 30 "Valkey on :$port" valkey-cli -p "$port" ping; done
  valkey-cli --cluster create 127.0.0.1:17000 127.0.0.1:17001 127.0.0.1:17002 \
    --cluster-replicas 0 --cluster-yes >"$LOGS/valkey-cluster-create.log" 2>&1
  wait_for 30 "the Redis cluster" redis_up
}
redis_stop() {
  local port
  for port in "${REDIS_PORTS[@]}"; do valkey-cli -p "$port" shutdown nosave >/dev/null 2>&1 || true; done
}

rabbitmq_up() { rabbitmq-diagnostics -q check_running >/dev/null 2>&1; }
rabbitmq_start() {
  if ! rabbitmq_up; then
    mkdir -p "$RABBITMQ_MNESIA_BASE"
    setsid rabbitmq-server >>"$LOGS/rabbitmq-server.out" 2>&1 < /dev/null &
    wait_for 120 RabbitMQ rabbitmq_up
  fi
  rabbitmqctl -q add_user "$RABBITMQ_DEFAULT_USER" "$RABBITMQ_DEFAULT_PASS" >/dev/null 2>&1 \
    || rabbitmqctl -q change_password "$RABBITMQ_DEFAULT_USER" "$RABBITMQ_DEFAULT_PASS" >/dev/null
  rabbitmqctl -q set_user_tags "$RABBITMQ_DEFAULT_USER" administrator >/dev/null
  rabbitmqctl -q set_permissions -p / "$RABBITMQ_DEFAULT_USER" '.*' '.*' '.*' >/dev/null
}
rabbitmq_stop() {
  if rabbitmq_up; then rabbitmqctl -q shutdown >/dev/null 2>&1 || true; fi
}

clamav_up() { echo PING | nc -w 2 127.0.0.1 3310 2>/dev/null | grep -q PONG; }
clamav_start() {
  clamav_up && return 0
  install -d /run/clamav
  clamd --config-file=/etc/autogpt-scanner/clamd.conf
  wait_for 60 ClamAV clamav_up
}
clamav_stop() {
  if [ -f /run/clamav/clamd.pid ]; then kill "$(cat /run/clamav/clamd.pid)" 2>/dev/null || true; fi
  rm -f /run/clamav/clamd.pid
}

usage() {
  sed -n '2,10s/^# \{0,1\}//p' "$0" >&2
  exit 2
}

command=${1:-}
[ $# -gt 0 ] && shift
services=("$@")
[ ${#services[@]} -gt 0 ] || services=(postgres redis rabbitmq clamav)
for service in "${services[@]}"; do
  case "$service" in postgres | redis | rabbitmq | clamav) ;; *) usage ;; esac
done

case "$command" in
  start)
    for service in "${services[@]}"; do "${service}_start"; echo "$service: up"; done
    ;;
  stop)
    for ((i = ${#services[@]} - 1; i >= 0; i--)); do "${services[i]}_stop"; echo "${services[i]}: stopped"; done
    ;;
  restart)
    "$0" stop "${services[@]}"
    "$0" start "${services[@]}"
    ;;
  status)
    down=0
    for service in "${services[@]}"; do
      if "${service}_up"; then echo "$service: up"; else echo "$service: down"; down=1; fi
    done
    exit "$down"
    ;;
  *) usage ;;
esac
EOF

COPY <<'EOF' /etc/rabbitmq/rabbitmq.conf
listeners.tcp.default = 127.0.0.1:5672
distribution.listener.interface = 127.0.0.1
log.file.level = info
EOF

# ClamAV with the limits CI gives it. The full signature database is hundreds of megabytes to download and over a
# gigabyte of memory once loaded, and no test needs it: the tests mock detections, and only need clean files to scan
# clean. The one signature here is the EICAR test file's, as given in ClamAV's documentation, so an upload can still
# be shown to be caught.
COPY <<'EOF' /etc/autogpt-scanner/clamd.conf
TCPSocket 3310
TCPAddr 127.0.0.1
DatabaseDirectory /var/lib/autogpt-services/clamav
LogFile /var/log/autogpt-services/clamd.log
PidFile /run/clamav/clamd.pid
StreamMaxLength 50M
MaxFileSize 100M
MaxScanSize 100M
MaxThreads 4
ReadTimeout 300
EOF
RUN install -d /var/lib/autogpt-services/clamav \
    && echo '44d88612fea8a8f36de82e1278abb02f:68:Eicar-Test-Signature' > /var/lib/autogpt-services/clamav/eicar.hdb

# RabbitMQ listens on IPv4 only, so the backend is pointed at 127.0.0.1: "localhost" makes it try ::1 first and
# log a connection error for every service it starts.
COPY <<'EOF' /etc/autogpt-scanner/common.env
# Connection settings for the services `autogpt-services start` runs, and the secrets the backend needs.
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
export DIRECT_URL=$DATABASE_URL
export REDIS_HOST=localhost REDIS_PORT=17000
export RABBITMQ_HOST=127.0.0.1
export SKILLS_CATALOG_PATH=/opt/skills-catalog
. /etc/autogpt-scanner/secrets.env
EOF

# Login shells reset PATH in /etc/profile, then read this.
COPY <<'EOF' /etc/profile.d/autogpt-scanner.sh
export PATH="/opt/poetry/bin:/usr/lib/postgresql/15/bin:/opt/rabbitmq/sbin:/opt/erlang/bin:$PATH"
EOF

COPY <<'EOF' /etc/autogpt-scanner/test.env
# . /etc/autogpt-scanner/test.env: the environment backend CI runs the tests in
# (.github/workflows/platform-backend-ci.yml), pointed at the services `autogpt-services start` runs.
. /etc/autogpt-scanner/common.env
export CI=true PLAIN_OUTPUT=True RUN_ENV=local PORT=8080 TRIAL_TEST_DATABASE=1
export EXECUTION_MANAGER_PORT=18002 EXECUTION_SCHEDULER_PORT=18003 DATABASE_API_PORT=18005
export AGENT_API_PORT=18006 NOTIFICATION_SERVICE_PORT=18007
EOF

COPY <<'EOF' /etc/autogpt-scanner/stack.env
# . /etc/autogpt-scanner/stack.env: the environment for running the platform locally (`poetry run rest` and the
# other entry points in the backend, `pnpm start` in the frontend), on the default ports the frontend build was
# made for. The backend behaves as the hosted service does (cloud mode, with credits), not as a local install, which
# grants every paid feature and skips some limits.
. /etc/autogpt-scanner/common.env
export PLATFORM_BASE_URL=http://localhost:8006 FRONTEND_BASE_URL=http://localhost:3000
export JWT_JWKS_URL=http://localhost:3000/api/auth/jwks
export BEHAVE_AS=cloud ENABLE_CREDIT=true
EOF

# Secrets for this image only, made when it is built: the backend refuses to start without the first two.
RUN { \
      echo "export ENCRYPTION_KEY=$(openssl rand -base64 32 | tr '+/' '-_')"; \
      echo "export UNSUBSCRIBE_SECRET_KEY=$(openssl rand -base64 32 | tr '+/' '-_')"; \
      echo "export RABBITMQ_DEFAULT_USER=rabbitmq_user_default"; \
      echo "export RABBITMQ_DEFAULT_PASS=$(openssl rand -hex 16)"; \
    } > /etc/autogpt-scanner/secrets.env

# A PostgreSQL cluster holding the database CI tests against, migrated to this checkout's schema; then a check
# that every service starts.
WORKDIR /src/autogpt_platform/backend
RUN install -d -o postgres -g postgres -m 0700 "$PGDATA" \
    && runuser -u postgres -- /usr/lib/postgresql/15/bin/initdb -D "$PGDATA" \
        --username=postgres --auth=trust --encoding=UTF8 --locale=C.UTF-8 >/dev/null \
    && echo "listen_addresses = 'localhost'" >> "$PGDATA/postgresql.conf" \
    && autogpt-services start postgres \
    && psql -h localhost -U postgres -d postgres -v ON_ERROR_STOP=1 -q -f /src/autogpt_platform/db/init/00-init.sql \
    && . /etc/autogpt-scanner/test.env \
    && poetry run prisma migrate deploy \
    && autogpt-services start \
    && autogpt-services status \
    && autogpt-services stop

WORKDIR /src
