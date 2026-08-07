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
  publish_readiness
  log "bootstrap complete"
}

wait_for_infrastructure() {
  wait_until "PostgreSQL" 180 \
    "${POSTGRES_BINDIR}/pg_isready" -q -h 127.0.0.1 -p 5432 -U postgres
  wait_until "Valkey node 17000" 120 "${PROBE[@]}" redis --port 17000
  wait_until "Valkey node 17001" 120 "${PROBE[@]}" redis --port 17001
  wait_until "Valkey node 17002" 120 "${PROBE[@]}" redis --port 17002
  wait_until "RabbitMQ" 240 run_rabbitmq_cli /opt/rabbitmq/sbin/rabbitmq-diagnostics -q ping
  wait_until "FalkorDB" 120 "${PROBE[@]}" redis --port 6380 \
    --password-env GRAPHITI_FALKORDB_PASSWORD
  if [[ "${AUTOGPT_ENABLE_CLAMAV:-true}" == true ]]; then
    wait_until "ClamAV" 300 "${PROBE[@]}" clam --port 3310
  fi
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
  if "${PROBE[@]}" redis --port 17000 --cluster >/dev/null 2>&1; then
    log "Valkey cluster is already healthy"
    return 0
  fi

  local known_nodes
  local fresh_cluster=true
  for port in 17000 17001 17002; do
    known_nodes="$(
      valkey-cli -h 127.0.0.1 -p "${port}" cluster info 2>/dev/null |
        awk -F: '$1 == "cluster_known_nodes" {gsub("\r", "", $2); print $2}'
    )"
    [[ "${known_nodes}" == "1" ]] || fresh_cluster=false
  done

  if [[ "${fresh_cluster}" == true ]]; then
    log "forming three-node Valkey cluster"
    valkey-cli --cluster create \
      127.0.0.1:17000 127.0.0.1:17001 127.0.0.1:17002 \
      --cluster-replicas 0 --cluster-yes >/dev/null
    wait_until "Valkey cluster" 60 "${PROBE[@]}" redis --port 17000 --cluster
    return 0
  fi

  # Existing nodes normally need only gossip time after all three processes
  # restart. Never destructively recreate a partially known cluster.
  local elapsed=0
  while ((elapsed < 90)); do
    if "${PROBE[@]}" redis --port 17000 --cluster >/dev/null 2>&1; then
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

  log "applying Prisma migrations"
  (
    cd "${AUTOGPT_BACKEND_DIR}"
    prisma migrate deploy
  )
}

publish_readiness() {
  local temporary
  temporary="$(mktemp "${AUTOGPT_RUNTIME_DIR}/ready.XXXXXX")"
  printf 'ready\n' >"${temporary}"
  chmod 0644 "${temporary}"
  mv -f "${temporary}" "${AUTOGPT_READY_FILE}"
}

main "$@"
