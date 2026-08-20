#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

readonly PROBE=("${AUTOGPT_BACKEND_DIR}/.venv/bin/python" "${AUTOGPT_ASSET_DIR}/probe.py")
readonly POSTGRES_BINDIR="${POSTGRES_BINDIR:-/usr/lib/postgresql/15/bin}"
readonly INIT_SQL="${AUTOGPT_DB_INIT_SQL:-${AUTOGPT_ASSET_DIR}/00-init.sql}"

main() {
  load_runtime_config
  rm -f "${AUTOGPT_READY_FILE}"
  wait_for_infrastructure
  ensure_valkey_cluster
  verify_rabbitmq_user
  migrate_database
  configure_frontend_database_role
  publish_readiness
  log "bootstrap complete"
}

wait_for_infrastructure() {
  wait_until "PostgreSQL" 180 \
    "${POSTGRES_BINDIR}/pg_isready" -q -h 127.0.0.1 -p 5432 -U postgres
  wait_until "Valkey node 17000" 120 "${PROBE[@]}" redis --port 17000 \
    --password-env REDIS_PASSWORD
  wait_until "Valkey node 17001" 120 "${PROBE[@]}" redis --port 17001 \
    --password-env REDIS_PASSWORD
  wait_until "Valkey node 17002" 120 "${PROBE[@]}" redis --port 17002 \
    --password-env REDIS_PASSWORD
  wait_until "RabbitMQ" 240 run_rabbitmq_cli /opt/rabbitmq/sbin/rabbitmq-diagnostics -q check_running
  wait_until "FalkorDB" 120 "${PROBE[@]}" redis --port 6380 \
    --password-env GRAPHITI_FALKORDB_PASSWORD
}

wait_until() {
  local description="$1"
  local timeout="$2"
  shift 2
  local elapsed=0
  until "$@" >/dev/null 2>&1; do
    ((elapsed < timeout)) || fatal "${description} was not ready within ${timeout}s"
    sleep 1
    ((elapsed += 1))
  done
  log "${description} is ready"
}

ensure_valkey_cluster() {
  if "${PROBE[@]}" redis --port 17000 --cluster \
    --password-env REDIS_PASSWORD >/dev/null 2>&1; then
    log "Valkey cluster is already healthy"
    return 0
  fi

  local known_nodes
  local fresh_cluster=true
  local port
  for port in 17000 17001 17002; do
    known_nodes="$(
      REDISCLI_AUTH="${REDIS_PASSWORD}" \
        valkey-cli -h 127.0.0.1 -p "${port}" cluster info 2>/dev/null |
        awk -F: '$1 == "cluster_known_nodes" {gsub("\r", "", $2); print $2}'
    )"
    [[ "${known_nodes}" == "1" ]] || fresh_cluster=false
  done

  if [[ "${fresh_cluster}" == true ]]; then
    log "forming three-node Valkey cluster"
    REDISCLI_AUTH="${REDIS_PASSWORD}" valkey-cli --cluster create \
      127.0.0.1:17000 127.0.0.1:17001 127.0.0.1:17002 \
      --cluster-replicas 0 --cluster-yes >/dev/null
    wait_until "Valkey cluster" 60 "${PROBE[@]}" redis --port 17000 --cluster \
      --password-env REDIS_PASSWORD
    return 0
  fi

  # Existing nodes normally need only gossip time after all three processes
  # restart. Never destructively recreate a partially known cluster.
  local elapsed=0
  while ((elapsed < 90)); do
    if "${PROBE[@]}" redis --port 17000 --cluster \
      --password-env REDIS_PASSWORD >/dev/null 2>&1; then
      log "Valkey cluster recovered"
      return 0
    fi
    sleep 1
    ((elapsed += 1))
  done
  fatal "Valkey cluster did not recover; refusing to erase persistent state"
}

verify_rabbitmq_user() {
  local users
  users="$(run_rabbitmq_cli /opt/rabbitmq/sbin/rabbitmqctl -q list_users)"
  awk '{print $1}' <<<"${users}" | grep -Fqx "${RABBITMQ_DEFAULT_USER}" || \
    fatal "RabbitMQ did not initialize the application user"
  if awk '{print $1}' <<<"${users}" | grep -Fqx guest; then
    fatal "RabbitMQ initialized the built-in guest user unexpectedly"
  fi
  log "RabbitMQ application user is present"
}

migrate_database() {
  [[ -f "${INIT_SQL}" ]] || fatal "database initialization SQL is missing: ${INIT_SQL}"
  log "ensuring platform database schemas exist"
  PGPASSWORD="${POSTGRES_PASSWORD}" \
    "${POSTGRES_BINDIR}/psql" \
    --host=127.0.0.1 \
    --port=5432 \
    --username=postgres \
    --dbname=postgres \
    --set=ON_ERROR_STOP=1 \
    --file="${INIT_SQL}" >/dev/null

  report_interrupted_migration

  log "applying Prisma migrations"
  (
    cd "${AUTOGPT_BACKEND_DIR}"
    prisma migrate deploy
  )
}

query_scalar() {
  PGPASSWORD="${POSTGRES_PASSWORD}"     "${POSTGRES_BINDIR}/psql"     --host=127.0.0.1     --port=5432     --username=postgres     --dbname=postgres     --set=ON_ERROR_STOP=1     --tuples-only     --no-align     --command="$1"
}

# Prisma records a migration before applying it and completes the row
# afterwards, so a container stopped during its first boot -- while
# `prisma migrate deploy` is still running -- can leave `finished_at` NULL.
# Every later boot then fails with Prisma's own "migration is in a failed
# state" trace, which says nothing about how the appliance got there or what to
# do next. Name the migration and the fix instead.
#
# Deliberately not resolved automatically: the interrupted migration may have
# applied part of its DDL, and marking it rolled back would skip the remainder
# on the next deploy, leaving a schema that matches neither state.
report_interrupted_migration() {
  local table_exists unfinished
  table_exists="$(query_scalar \
    "SELECT to_regclass('platform._prisma_migrations') IS NOT NULL")"
  [[ "${table_exists}" == t ]] || return 0

  unfinished="$(query_scalar "
    SELECT coalesce(string_agg(migration_name, ', ' ORDER BY started_at), '')
    FROM platform._prisma_migrations
    WHERE finished_at IS NULL AND rolled_back_at IS NULL")"
  [[ -n "${unfinished}" ]] || return 0

  log "a previous database migration did not finish: ${unfinished}"
  log "This usually means the container was stopped while it was still starting"
  log "for the first time."
  log ""
  log "If that was the first boot, no data exists yet: delete this container's"
  log "/data volume (on Unraid, its appdata directory) and start it again."
  log ""
  log "If the instance already holds data, restore it from a backup taken"
  log "before the update."
  log ""
  log "The migration can also be marked resolved by hand, but note that this"
  log "container will not stay up to run the command: it needs PostgreSQL"
  log "running against ${PGDATA:-/data/postgres} some other way. Determine"
  log "whether that migration's changes reached the database first, because the"
  log "two answers are not interchangeable:"
  log "  prisma migrate resolve --rolled-back ${unfinished}   # changes absent"
  log "  prisma migrate resolve --applied     ${unfinished}   # changes present"
  fatal "refusing to migrate over an interrupted migration"
}

configure_frontend_database_role() {
  log "configuring least-privilege frontend database role"
  PGPASSWORD="${POSTGRES_PASSWORD}" \
    "${POSTGRES_BINDIR}/psql" \
    --host=127.0.0.1 \
    --port=5432 \
    --username=postgres \
    --dbname=postgres \
    --set=ON_ERROR_STOP=1 \
    --quiet <<'SQL'
BEGIN;

DO $role$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'autogpt_frontend'
  ) THEN
    CREATE ROLE autogpt_frontend LOGIN;
  END IF;
END
$role$;

ALTER ROLE autogpt_frontend
  LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS
  CONNECTION LIMIT 10 PASSWORD NULL;
ALTER ROLE autogpt_frontend RESET ALL;
ALTER ROLE autogpt_frontend IN DATABASE postgres RESET ALL;

REVOKE ALL PRIVILEGES ON DATABASE postgres FROM autogpt_frontend;
REVOKE TEMPORARY ON DATABASE postgres FROM PUBLIC;
GRANT CONNECT ON DATABASE postgres TO autogpt_frontend;

REVOKE ALL PRIVILEGES ON SCHEMA platform FROM PUBLIC, autogpt_frontend;
GRANT USAGE ON SCHEMA platform TO autogpt_frontend;

REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA platform
  FROM PUBLIC, autogpt_frontend;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA platform
  REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;

REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA platform
  FROM PUBLIC, autogpt_frontend;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA platform
  FROM PUBLIC, autogpt_frontend;

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE
  platform."UserAuthIdentity",
  platform."UserAuthSession",
  platform."UserAuthAccount",
  platform."UserAuthVerification",
  platform."UserAuthJwks"
TO autogpt_frontend;

GRANT SELECT (id, email), UPDATE (email, "updatedAt")
  ON TABLE platform."User"
  TO autogpt_frontend;

DO $membership$
DECLARE
  frontend_role_oid oid;
BEGIN
  SELECT oid INTO STRICT frontend_role_oid
  FROM pg_catalog.pg_roles
  WHERE rolname = 'autogpt_frontend';

  IF EXISTS (
    SELECT 1
    FROM pg_catalog.pg_auth_members membership
    WHERE membership.member = frontend_role_oid
  ) THEN
    RAISE EXCEPTION 'autogpt_frontend must not belong to another database role';
  END IF;

  IF EXISTS (
    SELECT 1 FROM pg_catalog.pg_database WHERE datdba = frontend_role_oid
    UNION ALL
    SELECT 1 FROM pg_catalog.pg_namespace WHERE nspowner = frontend_role_oid
    UNION ALL
    SELECT 1 FROM pg_catalog.pg_class WHERE relowner = frontend_role_oid
    UNION ALL
    SELECT 1 FROM pg_catalog.pg_proc WHERE proowner = frontend_role_oid
  ) THEN
    RAISE EXCEPTION 'autogpt_frontend must not own database objects';
  END IF;
END
$membership$;

COMMIT;
SQL
}

publish_readiness() {
  local temporary
  temporary="$(mktemp "${AUTOGPT_RUNTIME_DIR}/ready.XXXXXX")"
  printf 'ready\n' >"${temporary}"
  chmod 0644 "${temporary}"
  mv -f "${temporary}" "${AUTOGPT_READY_FILE}"
}

main "$@"
