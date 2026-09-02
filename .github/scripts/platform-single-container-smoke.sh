#!/usr/bin/env bash

set -Eeuo pipefail

: "${SMOKE_IMAGE:?SMOKE_IMAGE is required}"
: "${SMOKE_PLATFORM:?SMOKE_PLATFORM is required}"

readonly PUBLIC_URL=http://localhost:3300
readonly EXPECTED_CODEX_TEMP_ROOT=/dev/shm/autogpt-codex
# Unraid ships Docker's stock stop timeout and operators must not have to
# raise it host-wide to run the appliance, so shutdown has to finish inside
# it. Stopping with a generous timeout would pass here while a default host
# SIGKILLs the container at 10s (exit 137) with the data stores still up.
readonly STOCK_DOCKER_STOP_TIMEOUT=10
# Exit code alone is a cliff: it only changes at the timeout, so a shutdown
# drifting from 6s to 9.9s passes identically to a fast one until it tips over
# to 137 on a slower host. Gate on the margin too, so erosion is caught here
# rather than by an operator. This is the same ceiling the unit test applies --
# DOCKER_STOP_TIMEOUT_SECONDS minus SHUTDOWN_MARGIN_SECONDS in
# single-container/tests/test_supervisor_config.py -- derived here rather than
# written down twice with different values.
readonly SHUTDOWN_MARGIN_SECONDS=1
readonly MAX_CLEAN_STOP_SECONDS=$((STOCK_DOCKER_STOP_TIMEOUT - SHUTDOWN_MARGIN_SECONDS))
readonly TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-2700}"
readonly SCAN_COMPLETION_FILE="${SMOKE_SCAN_COMPLETION_FILE:-}"
readonly SAFE_PLATFORM="${SMOKE_PLATFORM//\//-}"
readonly RUN_TOKEN="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-${SAFE_PLATFORM}-${RANDOM}"
readonly RUN_CONTAINER_NAME="autogpt-single-smoke-${RUN_TOKEN}"
readonly NEGATIVE_CONTAINER_NAME="${RUN_CONTAINER_NAME}-negative"
readonly SMOKE_STARTED_SECONDS=${SECONDS}
SMOKE_LAST_TIMING_SECONDS=${SECONDS}
HEADERS_FILE="$(mktemp)"
readonly HEADERS_FILE
AUTH_COOKIE_FILE="$(mktemp)"
readonly AUTH_COOKIE_FILE

CONTAINER_NAME="${RUN_CONTAINER_NAME}"
DATA_VOLUME=

record_timing() {
  local phase="$1"
  local now=${SECONDS}
  printf 'single-container smoke timing: phase=%s duration=%ss elapsed=%ss\n' \
    "${phase}" \
    "$((now - SMOKE_LAST_TIMING_SECONDS))" \
    "$((now - SMOKE_STARTED_SECONDS))"
  SMOKE_LAST_TIMING_SECONDS=${now}
}

record_boot_milestones() {
  local phase="$1"
  local started_at
  local health_history
  local container_logs
  local line
  local milestone
  local timestamp

  if ! started_at="$(
    docker inspect --format '{{.State.StartedAt}}' "${CONTAINER_NAME}"
  )"; then
    echo "could not record ${phase} container start time" >&2
    return 0
  fi
  printf 'single-container boot milestone: phase=%s container-started-at=%s\n' \
    "${phase}" "${started_at}"

  if health_history="$(
    docker inspect --format '{{json .State.Health.Log}}' "${CONTAINER_NAME}"
  )"; then
    if ! jq -r --arg phase "${phase}" \
      '.[] | "single-container boot milestone: phase=\($phase) docker-health start=\(.Start) end=\(.End) exit=\(.ExitCode)"' \
      <<<"${health_history}"; then
      echo "could not format ${phase} Docker health milestones" >&2
    fi
  else
    echo "could not record ${phase} Docker health milestones" >&2
  fi

  if ! container_logs="$(
    docker logs --timestamps --since "${started_at}" "${CONTAINER_NAME}" 2>&1
  )"; then
    echo "could not record ${phase} appliance milestones" >&2
    return 0
  fi
  # Raw service logs and Docker health output can contain request or provider
  # data and stay private. A matching line selects a fixed event name; none of
  # the original log content is emitted.
  while IFS= read -r line; do
    milestone=
    case "${line}" in
      *"[single-container] starting process supervisor"*) milestone=supervisor-start ;;
      *"[single-container] initializing PostgreSQL data directory"*) milestone=postgres-init ;;
      *"[single-container] PostgreSQL is ready"*) milestone=postgres-ready ;;
      *"[single-container] Valkey node "*" is ready"*) milestone=valkey-node-ready ;;
      *"[single-container] RabbitMQ is ready"*) milestone=rabbitmq-ready ;;
      *"[single-container] FalkorDB is ready"*) milestone=falkordb-ready ;;
      *"[single-container] forming three-node Valkey cluster"*) milestone=valkey-cluster-forming ;;
      *"[single-container] Valkey cluster is ready"*) milestone=valkey-cluster-ready ;;
      *"[single-container] Valkey cluster is already healthy"*) milestone=valkey-cluster-already-healthy ;;
      *"[single-container] Valkey cluster recovered"*) milestone=valkey-cluster-recovered ;;
      *"[single-container] RabbitMQ application user is present"*) milestone=rabbitmq-user-ready ;;
      *"[single-container] ensuring platform database schemas exist"*) milestone=database-schemas-start ;;
      *"[single-container] applying Prisma migrations"*) milestone=prisma-migrations-start ;;
      *"[single-container] configuring least-privilege frontend database role"*) milestone=frontend-role-start ;;
      *"[single-container] bootstrap complete"*) milestone=bootstrap-complete ;;
      *"[single-container] starting database-manager"*) milestone=database-manager-start ;;
      *"[single-container] starting scheduler"*) milestone=scheduler-start ;;
      *"[single-container] starting batch-executor"*) milestone=batch-executor-start ;;
      *"[single-container] starting notification"*) milestone=notification-start ;;
      *"[single-container] starting executor"*) milestone=executor-start ;;
      *"[single-container] starting copilot-executor"*) milestone=copilot-executor-start ;;
      *"[single-container] starting copilot-bot"*) milestone=copilot-bot-start ;;
      *"[single-container] starting platform-linking-manager"*) milestone=platform-linking-manager-start ;;
      *"[single-container] starting websocket"*) milestone=websocket-start ;;
      *"[single-container] starting rest"*) milestone=rest-start ;;
      *"[single-container] starting next"*) milestone=next-start ;;
      *"[single-container] starting nginx"*) milestone=nginx-start ;;
      *"[single-container] watchdog armed after initial healthy state"*) milestone=watchdog-armed ;;
    esac
    [[ -n "${milestone}" ]] || continue
    timestamp="${line%% *}"
    if [[ ! "${timestamp}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$ ]]; then
      timestamp=unknown
    fi
    printf 'single-container boot milestone: phase=%s at=%s event=%s\n' \
      "${phase}" "${timestamp}" "${milestone}"
  done <<<"${container_logs}"
}

diagnostics() {
  if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    if ! docker inspect --format '{{json .State}}' "${CONTAINER_NAME}"; then
      echo "could not inspect ${CONTAINER_NAME}" >&2
    fi
    if ! docker logs --timestamps --tail 2000 "${CONTAINER_NAME}"; then
      echo "could not read logs for ${CONTAINER_NAME}" >&2
    fi
  fi
}

cleanup() {
  local result=$?
  local cleanup_failed=0
  local container
  local container_id
  local volume_name
  trap - EXIT INT TERM
  if ((result != 0)); then
    diagnostics
  fi
  for container in "${CONTAINER_NAME}" "${NEGATIVE_CONTAINER_NAME}"; do
    if ! container_id="$(
      docker container ls --all --quiet --filter "name=^/${container}$"
    )"; then
      echo "could not determine whether ${container} needs cleanup" >&2
      cleanup_failed=1
      continue
    fi
    if [[ -n "${container_id}" ]]; then
      if ! docker stop --timeout "${STOCK_DOCKER_STOP_TIMEOUT}" "${container}" >/dev/null 2>&1; then
        echo "could not stop ${container} during cleanup" >&2
        cleanup_failed=1
      fi
      if ! docker rm --force --volumes "${container}" >/dev/null 2>&1; then
        echo "could not remove ${container} during cleanup" >&2
        cleanup_failed=1
      fi
    fi
  done
  if [[ -n "${DATA_VOLUME}" ]]; then
    if ! volume_name="$(
      docker volume ls --quiet --filter "name=^${DATA_VOLUME}$"
    )"; then
      echo "could not determine whether ${DATA_VOLUME} needs cleanup" >&2
      cleanup_failed=1
    elif [[ -n "${volume_name}" ]]; then
      if ! docker volume rm "${DATA_VOLUME}" >/dev/null 2>&1; then
        echo "could not remove ${DATA_VOLUME} during cleanup" >&2
        cleanup_failed=1
      fi
    fi
  fi
  if ! rm -f "${HEADERS_FILE}" "${AUTH_COOKIE_FILE}"; then
    echo "could not remove single-container smoke temporary files" >&2
    cleanup_failed=1
  fi
  if ((result == 0 && cleanup_failed != 0)); then
    result=1
  fi
  exit "${result}"
}

assert_clean_stop() {
  local reason="$1"
  local started elapsed exit_code
  # Integer SECONDS truncates at both ends, so a real 8.9s stop reads as 8.
  started="${EPOCHREALTIME}"
  docker stop --timeout "${STOCK_DOCKER_STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null
  elapsed="$(awk -v a="${started}" -v b="${EPOCHREALTIME}" 'BEGIN { printf "%.2f", b - a }')"
  exit_code="$(docker inspect --format '{{.State.ExitCode}}' "${CONTAINER_NAME}")"
  [[ "${exit_code}" == 0 ]] || {
    echo "container did not exit cleanly ${reason}: exit ${exit_code} after" \
      "${elapsed}s against the stock ${STOCK_DOCKER_STOP_TIMEOUT}s timeout" >&2
    return 1
  }
  awk -v e="${elapsed}" -v m="${MAX_CLEAN_STOP_SECONDS}" 'BEGIN { exit !(e <= m) }' || {
    echo "container exited cleanly ${reason} but took ${elapsed}s, over the" \
      "${MAX_CLEAN_STOP_SECONDS}s budget, close to the stock" \
      "${STOCK_DOCKER_STOP_TIMEOUT}s timeout" >&2
    return 1
  }
  echo "clean stop ${reason} in ${elapsed}s"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "${TIMEOUT_SECONDS}" =~ ^[0-9]+$ ]] || {
  echo "SMOKE_TIMEOUT_SECONDS must be an integer" >&2
  exit 2
}

appliance_is_healthy_and_armed() {
  docker exec "${CONTAINER_NAME}" \
    /usr/bin/test -f /run/autogpt/watchdog-armed >/dev/null 2>&1 &&
    docker exec "${CONTAINER_NAME}" \
      /usr/local/bin/autogpt-healthcheck >/dev/null 2>&1
}

wait_for_healthy() {
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  local state
  local health
  local exit_code
  local state_record
  while ((SECONDS < deadline)); do
    if ! state_record="$(
      docker inspect \
        --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}} {{.State.ExitCode}}' \
        "${CONTAINER_NAME}"
    )"; then
      echo "could not inspect container state while waiting for health" >&2
      return 1
    fi
    read -r state health exit_code <<<"${state_record}"
    case "${state}:${health}" in
      running:healthy)
        if appliance_is_healthy_and_armed; then
          return 0
        fi
        ;;
      exited:* | dead:* | removing:* | *:unhealthy)
        echo "container failed while waiting for health: state=${state} health=${health} exit=${exit_code}" >&2
        return 1
        ;;
    esac
    sleep 1
  done
  echo "container did not become healthy within ${TIMEOUT_SECONDS}s" >&2
  return 1
}

wait_for_automatic_restart() {
  local minimum_restart_count="$1"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  local state
  local health
  local restart_count
  local state_record
  while ((SECONDS < deadline)); do
    if ! state_record="$(
      docker inspect \
        --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}} {{.RestartCount}}' \
        "${CONTAINER_NAME}"
    )"; then
      echo "could not inspect container state while waiting for restart" >&2
      return 1
    fi
    read -r state health restart_count <<<"${state_record}"
    if [[ "${state}:${health}" == running:healthy ]] &&
      ((restart_count >= minimum_restart_count)) &&
      appliance_is_healthy_and_armed; then
      return 0
    fi
    case "${state}" in
      dead | removing)
        echo "container failed while waiting for automatic restart: state=${state}" >&2
        return 1
        ;;
    esac
    sleep 1
  done
  echo "container did not automatically restart and become healthy" >&2
  return 1
}

wait_for_concurrent_scans() {
  if [[ -z "${SCAN_COMPLETION_FILE}" ]]; then
    return 0
  fi
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  while ((SECONDS < deadline)); do
    if [[ -f "${SCAN_COMPLETION_FILE}" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "concurrent image scans did not complete within ${TIMEOUT_SECONDS}s" >&2
  return 1
}

count_container_log_evidence() {
  local expected="$1"
  local container_logs
  local count=0
  local line
  if ! container_logs="$(docker logs "${CONTAINER_NAME}" 2>&1)"; then
    echo "could not read container logs while counting watchdog evidence" >&2
    return 1
  fi
  while IFS= read -r line; do
    if [[ "${line}" == *"${expected}"* ]]; then
      count=$((count + 1))
    fi
  done <<<"${container_logs}"
  printf '%s\n' "${count}"
}

wait_for_new_container_log_evidence() {
  local expected="$1"
  local previous_count="$2"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  local current_count
  if [[ ! "${previous_count}" =~ ^[0-9]+$ ]]; then
    echo "previous watchdog evidence count is invalid" >&2
    return 1
  fi
  while ((SECONDS < deadline)); do
    if ! current_count="$(count_container_log_evidence "${expected}")"; then
      return 1
    fi
    if [[ ! "${current_count}" =~ ^[0-9]+$ ]]; then
      echo "watchdog evidence count is invalid" >&2
      return 1
    fi
    if ((current_count > previous_count)); then
      return 0
    fi
    sleep 1
  done
  echo "container log did not report new watchdog evidence: ${expected}" >&2
  return 1
}

force_watchdog_check() {
  local watchdog_pid="$1"
  local expected="$2"
  local previous_count
  if ! previous_count="$(count_container_log_evidence "${expected}")"; then
    return 1
  fi
  if [[ ! "${previous_count}" =~ ^[0-9]+$ ]]; then
    echo "watchdog evidence count is invalid before forced check" >&2
    return 1
  fi
  docker exec "${CONTAINER_NAME}" kill -USR1 "${watchdog_pid}"
  wait_for_new_container_log_evidence "${expected}" "${previous_count}"
}

runtime_config_hash() {
  docker exec "${CONTAINER_NAME}" sha256sum /data/config/runtime.env | awk '{print $1}'
}

discover_data_volume() {
  local mount_record
  local mount_type
  mount_record="$(
    docker inspect \
      --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Type}} {{.Name}}{{println}}{{end}}{{end}}' \
      "${CONTAINER_NAME}"
  )"
  read -r mount_type DATA_VOLUME <<<"${mount_record}"
  [[ "${mount_type}" == volume && "${DATA_VOLUME}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "image did not create one Docker volume at /data" >&2
    return 1
  }
}

preseed_hostile_backend_config() {
  docker run --rm \
    --platform "${SMOKE_PLATFORM}" \
    --volume "${DATA_VOLUME}:/data" \
    --entrypoint /app/autogpt_platform/backend/.venv/bin/python \
    "${SMOKE_IMAGE}" -c '
import json
import os
from pathlib import Path

path = Path("/data/config/backend.json")
path.write_text(json.dumps({
    "pyro_host": "0.0.0.0",
    "websocket_server_host": "0.0.0.0",
    "websocket_server_port": 18001,
    "execution_manager_port": 18002,
    "execution_scheduler_port": 18003,
    "database_api_port": 18005,
    "agent_api_host": "0.0.0.0",
    "agent_api_port": 18006,
    "notification_service_port": 18007,
    "copilot_executor_port": 18008,
    "platform_linking_service_port": 18009,
    "copilot_chat_bridge_port": 18010,
    "batch_executor_port": 18011,
}) + "\n", encoding="utf-8")
os.chown(path, 10001, 10001)
os.chmod(path, 0o600)
' >/dev/null
}

assert_pinned_topology_environment() {
  local pid
  local process_env
  pid="$(
    docker exec "${CONTAINER_NAME}" \
      supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf pid runtime:rest
  )"
  [[ "${pid}" =~ ^[0-9]+$ ]]
  # Docker Desktop can deny cross-UID /proc reads even to container root.
  # Read the service environment as the same unprivileged UID instead.
  process_env="$(
    docker exec --user autogpt "${CONTAINER_NAME}" \
      /bin/bash -Eeuo pipefail -c 'tr "\0" "\n" <"/proc/${1}/environ"' \
      bash "${pid}"
  )"
  for expected in \
    PYRO_HOST=127.0.0.1 \
    WEBSOCKET_SERVER_HOST=127.0.0.1 WEBSOCKET_SERVER_PORT=8001 \
    EXECUTION_MANAGER_PORT=8002 EXECUTION_SCHEDULER_PORT=8003 \
    DATABASE_API_PORT=8005 AGENT_API_HOST=127.0.0.1 AGENT_API_PORT=8006 \
    NOTIFICATION_SERVICE_PORT=8007 COPILOT_EXECUTOR_PORT=8008 \
    PLATFORM_LINKING_SERVICE_PORT=8009 COPILOT_CHAT_BRIDGE_PORT=8010 \
    BATCH_EXECUTOR_PORT=8011 FORCE_FLAG_GRAPHITI_MEMORY=true; do
    grep -Fxq "${expected}" <<<"${process_env}"
  done
}

assert_command_rejected_with() {
  local accepted_message="$1"
  local expected_message="$2"
  shift 2
  local output

  if output="$("$@" 2>&1)"; then
    echo "${accepted_message}" >&2
    return 1
  fi
  if [[ "${output,,}" != *"${expected_message,,}"* ]]; then
    echo "negative isolation probe failed for an unexpected reason" >&2
    echo "expected error containing: ${expected_message}" >&2
    printf '%s\n' "${output}" >&2
    return 1
  fi
}

assert_frontend_database_isolation() {
  local next_pid
  local nginx_pid
  local rest_pid
  local next_uid
  local nginx_uid
  local rest_uid
  local assertions
  local frontend_role_passwordless

  next_pid="$(
    docker exec "${CONTAINER_NAME}" \
      supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
      pid runtime:next
  )"
  rest_pid="$(
    docker exec "${CONTAINER_NAME}" \
      supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
      pid runtime:rest
  )"
  nginx_pid="$(
    docker exec "${CONTAINER_NAME}" \
      supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
      pid runtime:nginx
  )"
  [[ "${next_pid}" =~ ^[0-9]+$ && "${rest_pid}" =~ ^[0-9]+$ && \
    "${nginx_pid}" =~ ^[0-9]+$ ]]

  next_uid="$(docker exec "${CONTAINER_NAME}" stat -c '%u' "/proc/${next_pid}")"
  rest_uid="$(docker exec "${CONTAINER_NAME}" stat -c '%u' "/proc/${rest_pid}")"
  nginx_uid="$(docker exec "${CONTAINER_NAME}" stat -c '%u' "/proc/${nginx_pid}")"
  [[ "${next_uid}" == 10005 && "${nginx_uid}" == 10006 && \
    "${rest_uid}" == 10001 && "${next_uid}" != "${rest_uid}" && \
    "${nginx_uid}" != "${rest_uid}" ]] || {
    echo "public and backend process identities are not isolated" >&2
    return 1
  }

  docker exec --user autogpt_frontend "${CONTAINER_NAME}" \
    /bin/bash -Eeuo pipefail -c '
      [[ ! -r /data/config/runtime.env ]]
      [[ ! -r "/proc/${1}/environ" ]]
    ' bash "${rest_pid}"

  docker exec --user autogpt_proxy "${CONTAINER_NAME}" \
    /bin/bash -Eeuo pipefail -c '
      [[ ! -r /data/config/runtime.env ]]
      [[ ! -r "/proc/${1}/environ" ]]
      [[ ! -r "/proc/${2}/environ" ]]
    ' bash "${rest_pid}" "${next_pid}"

  docker exec --interactive --user autogpt_frontend "${CONTAINER_NAME}" \
    /app/autogpt_platform/backend/.venv/bin/python - "${next_pid}" <<'PY'
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

environment = {}
for item in Path(f"/proc/{sys.argv[1]}/environ").read_bytes().split(b"\0"):
    if item:
        name, value = item.split(b"=", 1)
        environment[name.decode("ascii")] = value.decode("utf-8")

forbidden = {
    "AUTH_DATABASE_URL",
    "DB_PASS",
    "DIRECT_URL",
    "ENCRYPTION_KEY",
    "GRAPHITI_FALKORDB_PASSWORD",
    "POSTGRES_PASSWORD",
    "RABBITMQ_DEFAULT_PASS",
    "REDIS_PASSWORD",
    "UNSUBSCRIBE_SECRET_KEY",
    "VAPID_PRIVATE_KEY",
}
assert forbidden.isdisjoint(environment)

database_url = urlsplit(environment["DATABASE_URL"])
query = parse_qs(database_url.query, strict_parsing=True)
assert database_url.scheme == "postgresql"
assert database_url.hostname is None
assert database_url.username is None
assert database_url.password is None
assert database_url.path == "/postgres"
assert query == {
    "host": ["/run/postgresql"],
    "user": ["autogpt_frontend"],
}
PY

  assertions="$(
    docker exec --interactive --user autogpt_frontend "${CONTAINER_NAME}" \
      /usr/bin/env -i \
      PATH=/usr/lib/postgresql/15/bin:/usr/bin:/bin \
      PGHOST=/run/postgresql \
      PGDATABASE=postgres \
      PGUSER=autogpt_frontend \
      psql --no-psqlrc --tuples-only --no-align --set=ON_ERROR_STOP=1 <<'SQL'
SELECT current_user = 'autogpt_frontend'
  AND NOT rolsuper
  AND NOT rolcreatedb
  AND NOT rolcreaterole
  AND NOT rolinherit
  AND NOT rolreplication
  AND NOT rolbypassrls
  AND rolconnlimit = 10
FROM pg_catalog.pg_roles
WHERE rolname = current_user;

SELECT count(*) = 0
FROM pg_catalog.pg_auth_members membership
JOIN pg_catalog.pg_roles role ON role.oid = membership.member
WHERE role.rolname = current_user;

SELECT has_database_privilege(current_user, 'postgres', 'CONNECT')
  AND NOT has_database_privilege(current_user, 'postgres', 'TEMPORARY')
  AND has_schema_privilege(current_user, 'platform', 'USAGE')
  AND NOT has_schema_privilege(current_user, 'platform', 'CREATE')
  AND NOT has_schema_privilege(current_user, 'public', 'CREATE')
  AND NOT has_schema_privilege(current_user, 'auth', 'USAGE');

SELECT COALESCE(
  bool_and(NOT has_function_privilege(current_user, function.oid, 'EXECUTE')),
  true
)
FROM pg_catalog.pg_proc function
JOIN pg_catalog.pg_namespace namespace ON namespace.oid = function.pronamespace
WHERE namespace.nspname = 'platform';

WITH required_table(name) AS (
  VALUES
    ('UserAuthIdentity'),
    ('UserAuthSession'),
    ('UserAuthAccount'),
    ('UserAuthVerification'),
    ('UserAuthJwks')
), required_privilege(name) AS (
  VALUES ('SELECT'), ('INSERT'), ('UPDATE'), ('DELETE')
)
SELECT bool_and(
  has_table_privilege(
    current_user,
    format('platform.%I', required_table.name),
    required_privilege.name
  )
)
FROM required_table CROSS JOIN required_privilege;

WITH required_table(name) AS (
  VALUES
    ('UserAuthIdentity'),
    ('UserAuthSession'),
    ('UserAuthAccount'),
    ('UserAuthVerification'),
    ('UserAuthJwks')
)
SELECT bool_and(
  NOT has_table_privilege(
    current_user,
    format('platform.%I', required_table.name),
    'TRUNCATE'
  )
)
FROM required_table;

SELECT has_column_privilege(current_user, 'platform."User"', 'id', 'SELECT')
  AND has_column_privilege(current_user, 'platform."User"', 'email', 'SELECT')
  AND has_column_privilege(current_user, 'platform."User"', 'email', 'UPDATE')
  AND has_column_privilege(current_user, 'platform."User"', 'updatedAt', 'UPDATE')
  AND NOT has_column_privilege(current_user, 'platform."User"', 'metadata', 'SELECT')
  AND NOT has_column_privilege(current_user, 'platform."User"', 'metadata', 'UPDATE')
  AND NOT has_table_privilege(current_user, 'platform."User"', 'SELECT')
  AND NOT has_table_privilege(current_user, 'platform."User"', 'UPDATE');
SQL
  )"
  [[ "${assertions}" == $'t\nt\nt\nt\nt\nt\nt' ]] || {
    echo "frontend database privileges are broader or narrower than expected" >&2
    printf 'database privilege assertions:\n%s\n' "${assertions}" >&2
    return 1
  }
  assert_command_rejected_with \
    "frontend database role can assume postgres" \
    'permission denied to set role "postgres"' \
    docker exec --user autogpt_frontend "${CONTAINER_NAME}" \
    /usr/bin/env -i \
    PATH=/usr/lib/postgresql/15/bin:/usr/bin:/bin \
    PGHOST=/run/postgresql \
    PGDATABASE=postgres \
    PGUSER=autogpt_frontend \
    psql --no-psqlrc --set=ON_ERROR_STOP=1 \
    --command='SET ROLE postgres'

  assert_command_rejected_with \
    "nginx operating-system user can access Valkey without authentication" \
    "noauth authentication required" \
    docker exec --user autogpt_proxy "${CONTAINER_NAME}" \
    /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    /app/autogpt_platform/backend/.venv/bin/python \
    /opt/autogpt/single-container/probe.py redis --port 17000

  frontend_role_passwordless="$(
    docker exec --user postgres "${CONTAINER_NAME}" \
      /usr/bin/env -i \
      PATH=/usr/lib/postgresql/15/bin:/usr/bin:/bin \
      PGHOST=/run/postgresql \
      PGDATABASE=postgres \
      PGUSER=postgres \
      psql --no-psqlrc --tuples-only --no-align --set=ON_ERROR_STOP=1 \
      --command="SELECT rolpassword IS NULL FROM pg_catalog.pg_authid WHERE rolname = 'autogpt_frontend'"
  )"
  [[ "${frontend_role_passwordless}" == t ]] || {
    echo "frontend database role unexpectedly has a password" >&2
    return 1
  }

  assert_command_rejected_with \
    "frontend operating-system user can authenticate as postgres" \
    'peer authentication failed for user "postgres"' \
    docker exec --user autogpt_frontend "${CONTAINER_NAME}" \
    /usr/bin/env -i \
    PATH=/usr/lib/postgresql/15/bin:/usr/bin:/bin \
    PGHOST=/run/postgresql \
    PGDATABASE=postgres \
    PGUSER=postgres \
    psql --no-psqlrc --set=ON_ERROR_STOP=1 \
    --command='SELECT 1'

  assert_command_rejected_with \
    "frontend operating-system user can access Valkey without authentication" \
    "noauth authentication required" \
    docker exec --user autogpt_frontend "${CONTAINER_NAME}" \
    /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    /app/autogpt_platform/backend/.venv/bin/python \
    /opt/autogpt/single-container/probe.py redis --port 17000

  docker exec "${CONTAINER_NAME}" \
    curl --fail --silent --show-error --max-time 30 \
    http://127.0.0.1:3001/api/auth/jwks >/dev/null
}

assert_email_auth_flow() {
  local email="single-container-${RUN_TOKEN}@example.com"
  local password=single-container-smoke-password # pragma: allowlist secret

  curl --fail-with-body --silent --show-error --max-time 30 \
    --cookie-jar "${AUTH_COOKIE_FILE}" \
    --request POST --header 'Content-Type: application/json' \
    --header "Origin: ${PUBLIC_URL}" \
    --data "{\"name\":\"Single Container Smoke\",\"email\":\"${email}\",\"password\":\"${password}\"}" \
    "${PUBLIC_URL}/api/auth/sign-up/email" >/dev/null

  curl --fail-with-body --silent --show-error --max-time 30 \
    --cookie "${AUTH_COOKIE_FILE}" \
    "${PUBLIC_URL}/api/auth/get-session" |
    python3 -c \
      'import json, sys; assert json.load(sys.stdin)["user"]["email"] == sys.argv[1]' \
      "${email}"

  curl --fail-with-body --silent --show-error --max-time 30 \
    --cookie "${AUTH_COOKIE_FILE}" --cookie-jar "${AUTH_COOKIE_FILE}" \
    --request POST --header 'Content-Type: application/json' \
    --header "Origin: ${PUBLIC_URL}" --data '{}' \
    "${PUBLIC_URL}/api/auth/sign-out" >/dev/null

  curl --fail-with-body --silent --show-error --max-time 30 \
    --cookie-jar "${AUTH_COOKIE_FILE}" \
    --request POST --header 'Content-Type: application/json' \
    --header "Origin: ${PUBLIC_URL}" \
    --data "{\"email\":\"${email}\",\"password\":\"${password}\"}" \
    "${PUBLIC_URL}/api/auth/sign-in/email" >/dev/null

  curl --fail-with-body --silent --show-error --max-time 30 \
    --cookie "${AUTH_COOKIE_FILE}" \
    "${PUBLIC_URL}/api/auth/get-session" |
    python3 -c \
      'import json, sys; assert json.load(sys.stdin)["user"]["email"] == sys.argv[1]' \
      "${email}"
}

assert_codex_provider_discovery() {
  curl --fail-with-body --silent --show-error --max-time 30 \
    --cookie "${AUTH_COOKIE_FILE}" \
    "${PUBLIC_URL}/api/proxy/api/integrations/providers" |
    python3 -c '
import json
import sys

providers = json.load(sys.stdin)
codex = [provider for provider in providers if provider.get("name") == "codex"]
assert len(codex) == 1, codex
assert "oauth2" in codex[0].get("supported_auth_types", []), codex[0]
'
}

assert_codex_runtime_contract() {
  local filesystem_type
  local ownership_and_mode
  local process_name
  local process_id
  local process_environment

  filesystem_type="$(
    docker exec "${CONTAINER_NAME}" \
      stat -f -c '%T' "${EXPECTED_CODEX_TEMP_ROOT}"
  )"
  [[ "${filesystem_type}" == tmpfs ]] || {
    echo "Codex temporary root is not memory-backed: ${filesystem_type}" >&2
    return 1
  }

  ownership_and_mode="$(
    docker exec "${CONTAINER_NAME}" \
      stat -c '%u:%g:%a' "${EXPECTED_CODEX_TEMP_ROOT}"
  )"
  [[ "${ownership_and_mode}" == 10001:10001:700 ]] || {
    echo "Codex temporary root is not owned by autogpt with mode 0700" >&2
    return 1
  }

  for process_name in runtime:rest runtime:executor runtime:copilot-executor; do
    process_id="$(
      docker exec "${CONTAINER_NAME}" \
        supervisorctl \
        -c /opt/autogpt/single-container/supervisor/supervisord.conf \
        pid "${process_name}"
    )"
    [[ "${process_id}" =~ ^[0-9]+$ ]]
    process_environment="$(
      docker exec --user autogpt "${CONTAINER_NAME}" \
        /bin/bash -Eeuo pipefail -c 'tr "\0" "\n" <"/proc/${1}/environ"' \
        bash "${process_id}"
    )"
    grep -Fxq \
      "CODEX_TEMP_ROOT=${EXPECTED_CODEX_TEMP_ROOT}" \
      <<<"${process_environment}" || {
      echo "${process_name} did not inherit CODEX_TEMP_ROOT" >&2
      return 1
    }
  done

  docker exec --interactive --user autogpt \
    --env "CODEX_TEMP_ROOT=${EXPECTED_CODEX_TEMP_ROOT}" \
    --workdir /app/autogpt_platform/backend \
    "${CONTAINER_NAME}" \
    /app/autogpt_platform/backend/.venv/bin/python - <<'PY'
import importlib.util
import os
from pathlib import Path

from backend.integrations.codex.http_client import API_BASE
from backend.integrations.codex.http_session import CodexHttpSession  # noqa: F401
from backend.integrations.oauth import DEVICE_HANDLERS_BY_NAME

root = Path(os.environ["CODEX_TEMP_ROOT"])
assert root == Path("/dev/shm/autogpt-codex")

# Codex reaches ChatGPT over HTTPS now. The bundled CLI was ~391 MB and the
# single biggest thing this image carried for it, so assert it is really
# absent rather than merely unused -- a transitive dependency could quietly
# drag it back in.
for banned in ("openai_codex", "codex_cli_bin"):
    assert importlib.util.find_spec(banned) is None, banned

assert API_BASE.startswith("https://"), API_BASE
assert "codex" in DEVICE_HANDLERS_BY_NAME, sorted(DEVICE_HANDLERS_BY_NAME)

print("codex-http-runtime-ok")
PY
}

assert_request_tokens_absent_from_logs() {
  local sentinel=AUTOGPT_LOG_SENTINEL_6f2b3cb87e9a
  local websocket_key=dGhlIHNhbXBsZSBub25jZQ== # pragma: allowlist secret # gitleaks:allow
  local websocket_status
  local container_logs

  curl --fail --silent --show-error --max-time 30 \
    "${PUBLIC_URL}/_agpt/health?token=${sentinel}" >/dev/null
  curl --silent --show-error --max-time 30 \
    "${PUBLIC_URL}/link/${sentinel}?token=${sentinel}" >/dev/null
  websocket_status="$(
    curl --silent --show-error --max-time 30 --http1.1 \
      --output /dev/null --write-out '%{http_code}' \
      --header 'Connection: Upgrade' \
      --header 'Upgrade: websocket' \
      --header 'Sec-WebSocket-Version: 13' \
      --header "Sec-WebSocket-Key: ${websocket_key}" \
      --header "Origin: ${PUBLIC_URL}" \
      "${PUBLIC_URL}/_agpt/ws?token=${sentinel}"
  )"
  [[ "${websocket_status}" == 403 ]] || {
    echo "invalid-token WebSocket handshake returned ${websocket_status}, expected 403" >&2
    return 1
  }
  sleep 2
  if ! container_logs="$(docker logs "${CONTAINER_NAME}" 2>&1)"; then
    echo "could not inspect container logs for request-token leakage" >&2
    return 1
  fi
  if [[ "${container_logs}" == *"${sentinel}"* ]]; then
    echo "request token sentinel leaked into container logs" >&2
    return 1
  fi
}

assert_internal_tooling_is_private() {
  local path
  local status
  for path in \
    /_agpt/docs /_agpt/redoc /_agpt/openapi.json /_agpt/metrics \
    /_agpt/external-api/docs /_agpt/external-api/redoc \
    /_agpt/external-api/openapi.json /_agpt/external-api/metrics; do
    status="$(
      curl --silent --show-error --max-time 30 \
        --output /dev/null --write-out '%{http_code}' "${PUBLIC_URL}${path}"
    )"
    [[ "${status}" == 404 ]] || {
      echo "internal tooling path ${path} returned ${status}, expected 404" >&2
      return 1
    }
  done
}

assert_runtime_config_mode() {
  local ownership_and_mode
  ownership_and_mode="$(
    docker exec "${CONTAINER_NAME}" stat -c '%u:%g:%a' /data/config/runtime.env
  )"
  [[ "${ownership_and_mode}" == 0:0:600 ]] || {
    echo "runtime config ownership/mode is not root:root 0600" >&2
    return 1
  }
}

assert_prisma_cli_is_prebundled() {
  local container_logs
  docker exec "${CONTAINER_NAME}" \
    /usr/bin/test -f \
    /opt/prisma-python/binaries/node_modules/prisma/build/index.js
  if ! container_logs="$(docker logs "${CONTAINER_NAME}" 2>&1)"; then
    echo "could not inspect container logs for runtime Prisma installation" >&2
    return 1
  fi
  if [[ "${container_logs}" == *"Installing Prisma CLI"* ]]; then
    echo "Prisma CLI was installed during container startup" >&2
    return 1
  fi
}

assert_falkordb_binary_contract() {
  local linkage
  local listeners
  local published_port
  linkage="$(
    docker exec "${CONTAINER_NAME}" /bin/bash -Eeuo pipefail -c '
      ldd /opt/falkordb/redis-server
      ldd /opt/falkordb/falkordb.so
    '
  )"
  if [[ "${linkage}" == *"not found"* ]]; then
    echo "FalkorDB has an unresolved shared-library dependency" >&2
    return 1
  fi

  listeners="$(docker exec "${CONTAINER_NAME}" ss -lnt)"
  [[ "${listeners}" =~ 127\.0\.0\.1:6380([[:space:]]|$) ]] || {
    echo "FalkorDB is not listening on the private loopback address" >&2
    return 1
  }
  if [[ "${listeners}" =~ (0\.0\.0\.0|\[::\]):6380([[:space:]]|$) ]]; then
    echo "FalkorDB is listening on a public container interface" >&2
    return 1
  fi
  if ! published_port="$(
    docker inspect \
      --format '{{with index .NetworkSettings.Ports "6380/tcp"}}{{json .}}{{end}}' \
      "${CONTAINER_NAME}"
  )"; then
    echo "could not inspect Docker port bindings for FalkorDB" >&2
    return 1
  fi
  if [[ -n "${published_port}" ]]; then
    echo "FalkorDB port 6380 is published by Docker" >&2
    return 1
  fi
}

assert_memory_contract() {
  local action="$1"
  docker exec --interactive \
    --workdir /app/autogpt_platform/backend \
    "${CONTAINER_NAME}" \
    /app/autogpt_platform/backend/.venv/bin/python - "${action}" <<'PY'
import asyncio
import shlex
import sys
from pathlib import Path

from falkordb import FalkorDB
from redis import Redis
from redis.exceptions import AuthenticationError


def runtime_value(name: str) -> str:
    for line in Path("/data/config/runtime.env").read_text().splitlines():
        if line.startswith(f"{name}="):
            values = shlex.split(line.split("=", 1)[1])
            if len(values) == 1:
                return values[0]
    raise AssertionError(f"missing {name} in runtime config")


password = runtime_value("GRAPHITI_FALKORDB_PASSWORD")
try:
    Redis(host="127.0.0.1", port=6380, password="wrong-password").ping()
except AuthenticationError:
    pass
else:
    raise AssertionError("FalkorDB accepted an invalid password")

redis = Redis(host="127.0.0.1", port=6380, password=password)
assert redis.ping()
assert "graph" in repr(redis.execute_command("MODULE", "LIST")).lower()
for setting, expected in {
    "MAX_QUEUED_QUERIES": "25",
    "TIMEOUT": "1000",
    "RESULTSET_SIZE": "10000",
}.items():
    configured = redis.execute_command("GRAPH.CONFIG", "GET", setting)
    assert expected in repr(configured), (setting, configured)

graph = FalkorDB(host="127.0.0.1", port=6380, password=password).select_graph(
    "autogpt_single_container_smoke"
)
action = sys.argv[1]
if action == "seed":
    graph.query("CREATE (:MemorySmoke {id: 'persistent-memory'})")

result = graph.query(
    "MATCH (n:MemorySmoke {id: 'persistent-memory'}) RETURN count(n)"
)
assert int(result.result_set[0][0]) == 1, result.result_set

from backend.copilot.graphiti.config import is_enabled_for_user

assert asyncio.run(is_enabled_for_user("single-container-smoke")) is True

if action == "cleanup":
    redis.execute_command("GRAPH.DELETE", graph.name)
PY
}

wait_for_falkordb_restart() {
  local previous_pid="$1"
  local deadline=$((SECONDS + TIMEOUT_SECONDS))
  local current_pid
  while ((SECONDS < deadline)); do
    if ! current_pid="$(
      docker exec "${CONTAINER_NAME}" \
        supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
        pid state:falkordb 2>/dev/null
    )"; then
      current_pid=""
    fi
    if [[ "${current_pid}" =~ ^[0-9]+$ ]] &&
      [[ "${current_pid}" != "${previous_pid}" ]] &&
      appliance_is_healthy_and_armed; then
      return 0
    fi
    sleep 1
  done
  echo "FalkorDB did not restart and return the container to healthy" >&2
  return 1
}

assert_redirect() {
  local request_url="$1"
  shift
  local status
  local location
  status="$(
    curl --silent --show-error --max-time 30 \
      --dump-header "${HEADERS_FILE}" --output /dev/null \
      --write-out '%{http_code}' "$@" "${request_url}"
  )"
  location="$(
    tr -d '\r' <"${HEADERS_FILE}" |
      awk 'tolower($1) == "location:" {sub(/^[^:]+:[[:space:]]*/, ""); print; exit}'
  )"
  [[ "${status}" == 307 ]] || {
    echo "protected route returned ${status}, expected 307" >&2
    return 1
  }
  [[ "${location}" == "${PUBLIC_URL}/login?next=%2Fcopilot" ]] || {
    echo "unexpected protected-route Location: ${location}" >&2
    return 1
  }
}

assert_prefixed_backend_redirect() {
  local status
  local location
  status="$(
    curl --silent --show-error --max-time 30 \
      --request POST --dump-header "${HEADERS_FILE}" --output /dev/null \
      --write-out '%{http_code}' "${PUBLIC_URL}/_agpt/api/email"
  )"
  location="$(
    tr -d '\r' <"${HEADERS_FILE}" |
      awk 'tolower($1) == "location:" {sub(/^[^:]+:[[:space:]]*/, ""); print; exit}'
  )"
  [[ "${status}" == 307 ]] || {
    echo "backend slash redirect returned ${status}, expected 307" >&2
    return 1
  }
  [[ "${location}" == "${PUBLIC_URL}/_agpt/api/email/" ]] || {
    echo "backend slash redirect lost its public prefix: ${location}" >&2
    return 1
  }
}

assert_unsupported_email_verification_rejected() {
  local output
  local status
  if output="$(
    timeout --signal=TERM --kill-after=30s 300s docker run --rm \
      --platform "${SMOKE_PLATFORM}" \
      --name "${NEGATIVE_CONTAINER_NAME}" \
      --env AUTH_REQUIRE_EMAIL_VERIFICATION=true \
      "${SMOKE_IMAGE}" 2>&1
  )"; then
    status=0
  else
    status=$?
  fi
  ((status != 0)) || {
    echo "image accepted unsupported email verification" >&2
    return 1
  }
  ((status != 124 && status != 137)) || {
    echo "email-verification rejection probe timed out" >&2
    return 1
  }
  [[ "${output}" == \
    *"email verification is not supported by the single-container distribution"* ]] || {
    echo "unsupported email verification failed without an actionable error" >&2
    return 1
  }
}

# Phase one deliberately supplies no environment, port, volume, entrypoint, or
# command override. This is the CI proof for literal `docker run IMAGE`.
assert_unsupported_email_verification_rejected
record_timing "unsupported-config-rejection"
docker run --detach \
  --platform "${SMOKE_PLATFORM}" \
  --name "${CONTAINER_NAME}" \
  "${SMOKE_IMAGE}" >/dev/null
discover_data_volume
wait_for_healthy
record_boot_milestones "initial-appliance-boot"
record_timing "initial-appliance-boot"
assert_codex_runtime_contract
assert_prisma_cli_is_prebundled
assert_falkordb_binary_contract
assert_memory_contract seed

first_hash="$(runtime_config_hash)"
[[ "${first_hash}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "runtime config checksum is invalid" >&2
  exit 1
}
assert_runtime_config_mode
assert_frontend_database_isolation

record_timing "phase-one-contracts"
assert_clean_stop "after docker stop"
docker rm "${CONTAINER_NAME}" >/dev/null
record_timing "phase-one-clean-stop"

# A persistent user config must not be able to move internal listeners or bind
# them publicly. Environment-pinned topology has higher Pydantic precedence.
preseed_hostile_backend_config

# Phase two proves explicit public-origin configuration, container replacement,
# and reuse of the exact state volume discovered above.
docker run --detach \
  --platform "${SMOKE_PLATFORM}" \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --publish 127.0.0.1:3300:3000 \
  --env "AUTOGPT_PUBLIC_URL=${PUBLIC_URL}" \
  --volume "${DATA_VOLUME}:/data" \
  "${SMOKE_IMAGE}" >/dev/null
wait_for_healthy
record_boot_milestones "replacement-appliance-boot"
record_timing "replacement-appliance-boot"

replacement_config_hash="$(runtime_config_hash)"
[[ "${replacement_config_hash}" == "${first_hash}" ]] || {
  echo "runtime config changed across container replacement" >&2
  exit 1
}
assert_runtime_config_mode
assert_frontend_database_isolation
assert_pinned_topology_environment
assert_email_auth_flow
assert_codex_provider_discovery
assert_falkordb_binary_contract
assert_memory_contract verify
curl --fail --silent --show-error "${PUBLIC_URL}/healthz" >/dev/null
assert_redirect "${PUBLIC_URL}/copilot" --resolve localhost:3300:127.0.0.1
assert_redirect http://127.0.0.1:3300/copilot \
  --header 'Host: attacker.invalid' \
  --header 'X-Forwarded-Proto: https'
assert_prefixed_backend_redirect
assert_internal_tooling_is_private
assert_request_tokens_absent_from_logs
record_timing "phase-two-contracts"

falkordb_pid="$(
  docker exec "${CONTAINER_NAME}" \
    supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
    pid state:falkordb
)"
[[ "${falkordb_pid}" =~ ^[0-9]+$ ]]
falkordb_restart_count="$(
  docker inspect --format '{{.RestartCount}}' "${CONTAINER_NAME}"
)"
docker exec "${CONTAINER_NAME}" kill -TERM "${falkordb_pid}"
wait_for_falkordb_restart "${falkordb_pid}"
record_timing "falkordb-process-recovery"
current_restart_count="$(
  docker inspect --format '{{.RestartCount}}' "${CONTAINER_NAME}"
)"
[[ "${current_restart_count}" == "${falkordb_restart_count}" ]] || {
  echo "FalkorDB recovery unexpectedly restarted the whole container" >&2
  exit 1
}
assert_memory_contract verify
wait_for_healthy

restart_count="$(docker inspect --format '{{.RestartCount}}' "${CONTAINER_NAME}")"
watchdog_pid="$(
  docker exec "${CONTAINER_NAME}" \
    supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
    pid runtime:watchdog
)"
[[ "${watchdog_pid}" =~ ^[0-9]+$ ]] || {
  echo "watchdog PID is invalid before forced health checks" >&2
  exit 1
}
# Reset the failure counter and the 30-second timer while every service is
# healthy, then prove each accelerated failure before requesting the next one.
force_watchdog_check \
  "${watchdog_pid}" \
  "watchdog health check passed trigger=forced"
record_timing "watchdog-forced-check-arm"

docker exec "${CONTAINER_NAME}" \
  supervisorctl \
  -c /opt/autogpt/single-container/supervisor/supervisord.conf \
  stop runtime:nginx >/dev/null
force_watchdog_check \
  "${watchdog_pid}" \
  "watchdog health failure 1/3 trigger=forced"
force_watchdog_check \
  "${watchdog_pid}" \
  "watchdog health failure 2/3 trigger=forced"
force_watchdog_check \
  "${watchdog_pid}" \
  "watchdog health failure 3/3 trigger=forced"
wait_for_automatic_restart "$((restart_count + 1))"
record_boot_milestones "automatic-appliance-restart"
record_timing "automatic-appliance-restart"
restarted_config_hash="$(runtime_config_hash)"
[[ "${restarted_config_hash}" == "${first_hash}" ]] || {
  echo "runtime config changed across automatic Docker restart" >&2
  exit 1
}
assert_runtime_config_mode
assert_frontend_database_isolation
assert_redirect "${PUBLIC_URL}/copilot" --resolve localhost:3300:127.0.0.1
assert_falkordb_binary_contract
assert_memory_contract cleanup

assert_clean_stop "after the restart test"
docker rm "${CONTAINER_NAME}" >/dev/null
record_timing "post-restart-contracts-and-clean-stop"
# Trivy scans the immutable image, independently of either runtime container.
# Synchronizing here preserves foreground scan failure/report propagation
# without holding phase one open before the replacement and restart checks.
wait_for_concurrent_scans
record_timing "final-scan-sync"

echo "single-container smoke test passed for ${SMOKE_PLATFORM}"
