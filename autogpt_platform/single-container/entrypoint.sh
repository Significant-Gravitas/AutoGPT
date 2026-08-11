#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

readonly AUTOGPT_PYTHON="${AUTOGPT_PYTHON:-${AUTOGPT_BACKEND_DIR}/.venv/bin/python}"
readonly POSTGRES_BINDIR="${POSTGRES_BINDIR:-/usr/lib/postgresql/15/bin}"

main() {
  [[ "$(id -u)" -eq 0 ]] || fatal "entrypoint must start as root so services can drop privileges"
  [[ -x "${AUTOGPT_PYTHON}" ]] || fatal "backend Python is missing: ${AUTOGPT_PYTHON}"

  prepare_directories
  initialize_backend_config
  "${AUTOGPT_PYTHON}" "${AUTOGPT_ASSET_DIR}/runtime_config.py" ensure \
    --path "${AUTOGPT_RUNTIME_ENV}"
  chown root:root "${AUTOGPT_RUNTIME_ENV}"
  chmod 0600 "${AUTOGPT_RUNTIME_ENV}"
  load_runtime_config
  configure_environment
  write_valkey_configs
  initialize_postgres
  write_rabbitmq_config
  write_falkordb_config
  rm -f "${AUTOGPT_READY_FILE}"

  if (($# == 0)); then
    set -- supervisord -c "${AUTOGPT_ASSET_DIR}/supervisor/supervisord.conf"
  fi
  log "starting process supervisor"
  exec "$@"
}

prepare_directories() {
  local managed_path
  for managed_path in \
    /data/config /data/postgres /data/rabbitmq /data/valkey \
    /data/valkey/17000 /data/valkey/17001 /data/valkey/17002 /data/falkordb \
    /data/workspaces /data/home /data/frontend-home \
    /data/cache /data/cache/backend /data/cache/next; do
    [[ ! -L "${managed_path}" ]] || fatal "refusing symlink at managed data path: ${managed_path}"
  done
  install -d -m 0710 -o root -g autogpt /data/config
  install -d -m 0700 -o postgres -g postgres /data/postgres
  install -d -m 0750 -o rabbitmq -g rabbitmq /data/rabbitmq
  install -d -m 0750 -o autogpt-valkey -g autogpt-valkey /data/valkey
  install -d -m 0750 -o autogpt-valkey -g autogpt-valkey /data/valkey/17000
  install -d -m 0750 -o autogpt-valkey -g autogpt-valkey /data/valkey/17001
  install -d -m 0750 -o autogpt-valkey -g autogpt-valkey /data/valkey/17002
  install -d -m 0750 -o autogpt-falkor -g autogpt-falkor /data/falkordb
  install -d -m 0750 -o autogpt -g autogpt /data/workspaces
  install -d -m 0750 -o autogpt -g autogpt /data/home
  install -d -m 0700 -o autogpt_frontend -g autogpt_frontend /data/frontend-home
  install -d -m 0711 -o root -g root /data/cache
  install -d -m 0750 -o autogpt -g autogpt /data/cache/backend
  install -d -m 0700 -o autogpt_frontend -g autogpt_frontend /data/cache/next
  install -d -m 0755 -o postgres -g postgres /run/postgresql
  # Service-specific runtime directories and files carry the restrictive
  # permissions. Keep only execute permission on their common parent so the
  # unprivileged PostgreSQL, RabbitMQ, FalkorDB, and app users can traverse to
  # their own assets without being able to list or modify sibling state.
  install -d -m 0711 -o root -g root "${AUTOGPT_RUNTIME_DIR}"
  install -d -m 0700 -o rabbitmq -g rabbitmq "${AUTOGPT_RUNTIME_DIR}/rabbitmq"
  install -d -m 0711 -o root -g root "${AUTOGPT_RUNTIME_DIR}/valkey"
  install -d -m 0700 -o autogpt_proxy -g autogpt_proxy "${AUTOGPT_RUNTIME_DIR}/nginx"
  install -d -m 0700 -o autogpt_proxy -g autogpt_proxy "${AUTOGPT_RUNTIME_DIR}/nginx/home"
  install -d -m 0700 -o autogpt_proxy -g autogpt_proxy "${AUTOGPT_RUNTIME_DIR}/nginx/cache"
  install -d -m 0700 -o autogpt_proxy -g autogpt_proxy "${AUTOGPT_RUNTIME_DIR}/nginx/client"
  install -d -m 0700 -o autogpt_proxy -g autogpt_proxy "${AUTOGPT_RUNTIME_DIR}/nginx/proxy"
  install -d -m 0700 -o autogpt_proxy -g autogpt_proxy "${AUTOGPT_RUNTIME_DIR}/nginx/fastcgi"
  install -d -m 0700 -o autogpt_proxy -g autogpt_proxy "${AUTOGPT_RUNTIME_DIR}/nginx/uwsgi"
  install -d -m 0700 -o autogpt_proxy -g autogpt_proxy "${AUTOGPT_RUNTIME_DIR}/nginx/scgi"
}

initialize_backend_config() {
  local path=/data/config/backend.json
  [[ ! -L "${path}" ]] || fatal "refusing symlink at backend config path"
  if [[ ! -e "${path}" ]]; then
    printf '{}\n' >"${path}"
  fi
  [[ -f "${path}" ]] || fatal "backend config must be a regular file"
  chown autogpt:autogpt "${path}"
  chmod 0600 "${path}"
  "${AUTOGPT_PYTHON}" -m json.tool "${path}" >/dev/null || \
    fatal "backend config is not valid JSON"
}

configure_environment() {
  validate_legacy_auth
  normalize_toggle AUTOGPT_ENABLE_BOT_SERVICES false
  if [[ "${AUTH_REQUIRE_EMAIL_VERIFICATION:-false}" != false ]]; then
    fatal "email verification is not supported by the single-container distribution"
  fi
  normalize_integer DB_CONNECTION_LIMIT 5 1 5
  normalize_integer DB_CONNECT_TIMEOUT 60 1 600
  normalize_integer DB_POOL_TIMEOUT 300 1 3600

  AUTOGPT_PUBLIC_URL="$(
    "${AUTOGPT_PYTHON}" "${AUTOGPT_ASSET_DIR}/runtime_config.py" \
      validate-public-url "${AUTOGPT_PUBLIC_URL:-http://localhost:3000}"
  )"
  export AUTOGPT_PUBLIC_URL
  log "public URL: ${AUTOGPT_PUBLIC_URL}"
  configure_account_registration
  write_nginx_public_url_config

  export PGDATA=/data/postgres
  export POSTGRES_USER=postgres POSTGRES_DB=postgres
  export DB_HOST=127.0.0.1 DB_PORT=5432 DB_USER=postgres DB_NAME=postgres
  export DB_PASS="${POSTGRES_PASSWORD}" DB_SCHEMA=platform
  export DATABASE_URL="postgresql://postgres:${POSTGRES_PASSWORD}@127.0.0.1:5432/postgres?schema=platform&connection_limit=${DB_CONNECTION_LIMIT}&connect_timeout=${DB_CONNECT_TIMEOUT}&pool_timeout=${DB_POOL_TIMEOUT}"
  # DIRECT_URL is also consumed by APScheduler's SQLAlchemy/psycopg2 job
  # store. Keep Prisma-only pool parameters on DATABASE_URL so that the
  # standard PostgreSQL driver never receives unsupported DSN options.
  export DIRECT_URL="postgresql://postgres:${POSTGRES_PASSWORD}@127.0.0.1:5432/postgres?schema=platform&connect_timeout=${DB_CONNECT_TIMEOUT}"
  export PRISMA_SCHEMA="${AUTOGPT_BACKEND_DIR}/schema.prisma" AUTH_DB_SCHEMA=platform

  export REDIS_HOST=127.0.0.1 REDIS_PORT=17000 REDIS_PASSWORD
  export REDIS_CLUSTER_HOST=127.0.0.1 REDIS_CLUSTER_PORT=17000
  export REDIS_USE_ANNOUNCED_ADDRESS=false
  export RABBITMQ_HOST=127.0.0.1 RABBITMQ_PORT=5672
  export RABBITMQ_CLUSTER_HOST=127.0.0.1 RABBITMQ_CLUSTER_PORT=5672
  export RABBITMQ_VHOST=/
  export RABBITMQ_MNESIA_BASE=/data/rabbitmq/mnesia
  export RABBITMQ_NODENAME=rabbit@localhost
  export RABBITMQ_CONFIG_FILE="${AUTOGPT_RUNTIME_DIR}/rabbitmq/rabbitmq"
  export GRAPHITI_FALKORDB_HOST=127.0.0.1 GRAPHITI_FALKORDB_PORT=6380
  # The appliance bundles no antivirus daemon. Force the scanner off so uploads
  # short-circuit as clean instead of failing on an unreachable ClamAV service.
  export CLAMAV_SERVICE_ENABLED=false

  export PYRO_HOST=127.0.0.1
  export AGENTSERVER_HOST=127.0.0.1 SCHEDULER_HOST=127.0.0.1
  export DATABASEMANAGER_HOST=127.0.0.1 EXECUTIONMANAGER_HOST=127.0.0.1
  export NOTIFICATIONMANAGER_HOST=127.0.0.1 PLATFORMLINKINGMANAGER_HOST=127.0.0.1
  export COPILOTEXECUTOR_HOST=127.0.0.1 COPILOTCHATBRIDGE_HOST=127.0.0.1
  export AGENT_API_HOST=127.0.0.1 WEBSOCKET_SERVER_HOST=127.0.0.1
  # The persistent backend JSON contains user-tunable product settings, but it
  # must not be able to move or expose appliance services. Environment values
  # have higher Pydantic priority, so pin the fixed internal topology here.
  export WEBSOCKET_SERVER_PORT="${AUTOGPT_WEBSOCKET_PORT}"
  export EXECUTION_MANAGER_PORT="${AUTOGPT_EXECUTION_MANAGER_PORT}"
  export EXECUTION_SCHEDULER_PORT="${AUTOGPT_EXECUTION_SCHEDULER_PORT}"
  export DATABASE_API_PORT="${AUTOGPT_DATABASE_API_PORT}"
  export AGENT_API_PORT="${AUTOGPT_AGENT_API_PORT}"
  export NOTIFICATION_SERVICE_PORT="${AUTOGPT_NOTIFICATION_SERVICE_PORT}"
  export COPILOT_EXECUTOR_PORT="${AUTOGPT_COPILOT_EXECUTOR_PORT}"
  export PLATFORM_LINKING_SERVICE_PORT="${AUTOGPT_PLATFORM_LINKING_SERVICE_PORT}"
  export COPILOT_CHAT_BRIDGE_PORT="${AUTOGPT_COPILOT_CHAT_BRIDGE_PORT}"
  export BATCH_EXECUTOR_PORT="${AUTOGPT_BATCH_EXECUTOR_PORT}"
  # Keep self-hosted product behavior without enabling LOCAL-only API docs and
  # asyncio debug mode on the public REST process.
  export APP_ENV=dev BEHAVE_AS=local ENABLE_AUTH=true

  export BETTER_AUTH_URL="${AUTOGPT_PUBLIC_URL}"
  export BETTER_AUTH_INTERNAL_URL=http://127.0.0.1:3001
  export JWT_JWKS_URL=http://127.0.0.1:3001/api/auth/jwks
  export FRONTEND_BASE_URL="${AUTOGPT_PUBLIC_URL}"
  export PLATFORM_BASE_URL="${AUTOGPT_PUBLIC_URL}/_agpt"
  export PLATFORM_LINK_BASE_URL="${AUTOGPT_PUBLIC_URL}/link"
  export AGPT_SERVER_URL="http://127.0.0.1:${AUTOGPT_AGENT_API_PORT}/api"
  export AGPT_WS_SERVER_URL="ws://127.0.0.1:${AUTOGPT_WEBSOCKET_PORT}/ws"
  export WORKSPACE_STORAGE_DIR=/data/workspaces
  export VAPID_CLAIM_EMAIL="${VAPID_CLAIM_EMAIL:-mailto:admin@localhost}"
  export AUTH_REQUIRE_EMAIL_VERIFICATION=false
  export NODE_ENV=production
  # Python imports this directory's sitecustomize module before each service
  # entry point, suppressing HTTP access targets and redacting WS query tokens.
  export PYTHONPATH="${AUTOGPT_ASSET_DIR}/python"
}

configure_account_registration() {
  normalize_toggle AUTH_ALLOW_NEW_ACCOUNTS true
  if [[ "${AUTH_ALLOW_NEW_ACCOUNTS}" == true ]]; then
    log "WARNING: open account registration is enabled; anyone who can reach the app can sign up"
    log "after creating the intended accounts, set AUTH_ALLOW_NEW_ACCOUNTS=false and recreate the container"
  else
    log "account registration is closed; temporarily set AUTH_ALLOW_NEW_ACCOUNTS=true to create intended accounts"
    log "after signup, run 'docker exec <container> autogpt-admin promote <email>', disable registration, and restart"
  fi
}

write_nginx_public_url_config() {
  local path="${AUTOGPT_RUNTIME_DIR}/nginx/public-url.conf"
  local public_scheme="${AUTOGPT_PUBLIC_URL%%://*}"
  local public_host="${AUTOGPT_PUBLIC_URL#*://}"
  [[ ! -L "${path}" ]] || fatal "refusing symlink at nginx public URL config"
  install -m 0600 -o autogpt_proxy -g autogpt_proxy /dev/null "${path}"
  {
    printf "set \$autogpt_public_url \"%s\";\n" "${AUTOGPT_PUBLIC_URL}"
    printf "set \$autogpt_public_host \"%s\";\n" "${public_host}"
    printf "set \$autogpt_public_scheme \"%s\";\n" "${public_scheme}"
  } >"${path}"
}

normalize_integer() {
  local name="$1"
  local default="$2"
  local minimum="$3"
  local maximum="$4"
  local value="${!name:-${default}}"
  [[ "${value}" =~ ^[0-9]+$ ]] || fatal "${name} must be an integer"
  ((value >= minimum && value <= maximum)) || \
    fatal "${name} must be between ${minimum} and ${maximum}"
  printf -v "${name}" '%s' "${value}"
  export "${name?}"
}

normalize_toggle() {
  local name="$1"
  local default="$2"
  local value="${!name:-${default}}"
  case "${value}" in
    true | false) ;;
    *) fatal "${name} must be true or false" ;;
  esac
  printf -v "${name}" '%s' "${value}"
  export "${name?}"
}

initialize_postgres() {
  [[ -x "${POSTGRES_BINDIR}/initdb" ]] || fatal "PostgreSQL 15 initdb is missing"
  [[ ! -e "${PGDATA}/PG_VERSION" || -f "${PGDATA}/PG_VERSION" ]] || \
    fatal "invalid PostgreSQL data directory"
  [[ ! -f "${PGDATA}/PG_VERSION" ]] || return 0

  log "initializing PostgreSQL data directory"
  local password_file
  password_file="$(mktemp "${AUTOGPT_RUNTIME_DIR}/postgres-password.XXXXXX")"
  trap 'rm -f "${password_file}"' RETURN
  printf '%s\n' "${POSTGRES_PASSWORD}" >"${password_file}"
  chown postgres:postgres "${password_file}"
  chmod 0600 "${password_file}"
  run_as postgres "${POSTGRES_BINDIR}/initdb" \
    --pgdata="${PGDATA}" \
    --username=postgres \
    --pwfile="${password_file}" \
    --auth-local=peer \
    --auth-host=scram-sha-256 \
    --encoding=UTF8 \
    --locale=C.UTF-8

  {
    printf "\nlisten_addresses = '127.0.0.1'\n"
    printf "port = 5432\n"
    printf "unix_socket_directories = '/run/postgresql'\n"
    printf "password_encryption = 'scram-sha-256'\n" # pragma: allowlist secret
  } >>"${PGDATA}/postgresql.conf"
  rm -f "${password_file}"
  trap - RETURN
}

write_valkey_configs() {
  local port
  local target
  local temporary
  for port in 17000 17001 17002; do
    target="${AUTOGPT_RUNTIME_DIR}/valkey/${port}.conf"
    [[ ! -L "${target}" ]] || fatal "refusing symlink at Valkey config path"
    temporary="$(mktemp "${AUTOGPT_RUNTIME_DIR}/valkey/.${port}.conf.XXXXXX")"
    trap 'rm -f "${temporary}"' RETURN
    chmod 0600 "${temporary}"
    {
      printf 'bind 127.0.0.1\n'
      printf 'protected-mode yes\n'
      printf 'port %s\n' "${port}"
      printf 'dir /data/valkey/%s\n' "${port}"
      printf 'appendonly yes\n'
      printf 'cluster-enabled yes\n'
      printf 'cluster-config-file nodes.conf\n'
      printf 'cluster-node-timeout 5000\n'
      printf 'cluster-require-full-coverage no\n'
      printf 'cluster-announce-ip 127.0.0.1\n'
      printf 'cluster-announce-port %s\n' "${port}"
      printf 'cluster-announce-bus-port %s\n' "$((port + 10000))"
      printf 'requirepass %s\n' "${REDIS_PASSWORD}"
      printf 'masterauth %s\n' "${REDIS_PASSWORD}"
    } >"${temporary}"
    chown autogpt-valkey:autogpt-valkey "${temporary}"
    chmod 0400 "${temporary}"
    mv -f "${temporary}" "${target}"
    trap - RETURN
  done
}

write_rabbitmq_config() {
  local temporary
  temporary="$(mktemp "${AUTOGPT_RUNTIME_DIR}/rabbitmq/rabbitmq.conf.XXXXXX")"
  trap 'rm -f "${temporary}"' RETURN
  chmod 0600 "${temporary}"
  {
    cat "${AUTOGPT_ASSET_DIR}/rabbitmq/rabbitmq.conf"
    printf 'default_user = %s\n' "${RABBITMQ_DEFAULT_USER}"
    printf 'default_pass = %s\n' "${RABBITMQ_DEFAULT_PASS}"
    printf 'default_vhost = /\n'
  } >"${temporary}"
  chown rabbitmq:rabbitmq "${temporary}"
  mv -f "${temporary}" "${AUTOGPT_RUNTIME_DIR}/rabbitmq/rabbitmq.conf"
  trap - RETURN
}

write_falkordb_config() {
  local temporary
  temporary="$(mktemp "${AUTOGPT_RUNTIME_DIR}/falkordb.conf.XXXXXX")"
  trap 'rm -f "${temporary}"' RETURN
  chmod 0600 "${temporary}"
  {
    printf 'bind 127.0.0.1\n'
    printf 'protected-mode yes\n'
    printf 'port 6380\n'
    printf 'daemonize no\n'
    printf 'dir /data/falkordb\n'
    printf 'appendonly yes\n'
    printf 'requirepass %s\n' "${GRAPHITI_FALKORDB_PASSWORD}"
    printf '%s\n' \
      'loadmodule /opt/falkordb/falkordb.so MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000'
  } >"${temporary}"
  chown autogpt-falkor:autogpt-falkor "${temporary}"
  mv -f "${temporary}" "${AUTOGPT_RUNTIME_DIR}/falkordb.conf"
  trap - RETURN
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
