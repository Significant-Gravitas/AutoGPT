#!/usr/bin/env bash

set -Eeuo pipefail

: "${SMOKE_IMAGE:?SMOKE_IMAGE is required}"
: "${SMOKE_PLATFORM:?SMOKE_PLATFORM is required}"

readonly PUBLIC_URL=http://localhost:3300
readonly EXPECTED_CODEX_TEMP_ROOT=/dev/shm/autogpt-codex
readonly TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-2700}"
readonly SAFE_PLATFORM="${SMOKE_PLATFORM//\//-}"
readonly RUN_TOKEN="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-${SAFE_PLATFORM}-${RANDOM}"
readonly RUN_CONTAINER_NAME="autogpt-single-smoke-${RUN_TOKEN}"
readonly NEGATIVE_CONTAINER_NAME="${RUN_CONTAINER_NAME}-negative"
HEADERS_FILE="$(mktemp)"
readonly HEADERS_FILE
AUTH_COOKIE_FILE="$(mktemp)"
readonly AUTH_COOKIE_FILE

CONTAINER_NAME="${RUN_CONTAINER_NAME}"
DATA_VOLUME=

diagnostics() {
  if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker inspect --format '{{json .State}}' "${CONTAINER_NAME}" || true
    docker logs --timestamps --tail 2000 "${CONTAINER_NAME}" || true
  fi
}

cleanup() {
  local result=$?
  local container
  trap - EXIT INT TERM
  if ((result != 0)); then
    diagnostics
  fi
  for container in "${CONTAINER_NAME}" "${NEGATIVE_CONTAINER_NAME}"; do
    if docker container inspect "${container}" >/dev/null 2>&1; then
      docker stop --timeout 360 "${container}" >/dev/null 2>&1 || true
      docker rm --force --volumes "${container}" >/dev/null 2>&1 || true
    fi
  done
  if [[ -n "${DATA_VOLUME}" ]] && docker volume inspect "${DATA_VOLUME}" >/dev/null 2>&1; then
    docker volume rm "${DATA_VOLUME}" >/dev/null 2>&1 || true
  fi
  rm -f "${HEADERS_FILE}" "${AUTH_COOKIE_FILE}"
  exit "${result}"
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
  while ((SECONDS < deadline)); do
    read -r state health exit_code < <(
      docker inspect \
        --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}} {{.State.ExitCode}}' \
        "${CONTAINER_NAME}"
    )
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
    sleep 10
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
  while ((SECONDS < deadline)); do
    read -r state health restart_count < <(
      docker inspect \
        --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}} {{.RestartCount}}' \
        "${CONTAINER_NAME}"
    )
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
    sleep 10
  done
  echo "container did not automatically restart and become healthy" >&2
  return 1
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
      supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf pid rest
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

assert_frontend_database_isolation() {
  local next_pid
  local nginx_pid
  local rest_pid
  local next_uid
  local nginx_uid
  local rest_uid
  local assertions

  next_pid="$(
    docker exec "${CONTAINER_NAME}" \
      supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
      pid next
  )"
  rest_pid="$(
    docker exec "${CONTAINER_NAME}" \
      supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
      pid rest
  )"
  nginx_pid="$(
    docker exec "${CONTAINER_NAME}" \
      supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
      pid nginx
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
  if docker exec --user autogpt_frontend "${CONTAINER_NAME}" \
    /usr/bin/env -i \
    PATH=/usr/lib/postgresql/15/bin:/usr/bin:/bin \
    PGHOST=/run/postgresql \
    PGDATABASE=postgres \
    PGUSER=autogpt_frontend \
    psql --no-psqlrc --set=ON_ERROR_STOP=1 \
    --command='SET ROLE postgres' >/dev/null 2>&1; then
    echo "frontend database role can assume postgres" >&2
    return 1
  fi

  if docker exec --user autogpt_proxy "${CONTAINER_NAME}" \
    /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    /app/autogpt_platform/backend/.venv/bin/python \
    /opt/autogpt/single-container/probe.py redis --port 17000 \
    >/dev/null 2>&1; then
    echo "nginx operating-system user can access Valkey without authentication" >&2
    return 1
  fi

  [[ "$(
    docker exec --user postgres "${CONTAINER_NAME}" \
      /usr/bin/env -i \
      PATH=/usr/lib/postgresql/15/bin:/usr/bin:/bin \
      PGHOST=/run/postgresql \
      PGDATABASE=postgres \
      PGUSER=postgres \
      psql --no-psqlrc --tuples-only --no-align --set=ON_ERROR_STOP=1 \
      --command="SELECT rolpassword IS NULL FROM pg_catalog.pg_authid WHERE rolname = 'autogpt_frontend'"
  )" == t ]] || {
    echo "frontend database role unexpectedly has a password" >&2
    return 1
  }

  if docker exec --user autogpt_frontend "${CONTAINER_NAME}" \
    /usr/bin/env -i \
    PATH=/usr/lib/postgresql/15/bin:/usr/bin:/bin \
    PGHOST=/run/postgresql \
    PGDATABASE=postgres \
    PGUSER=postgres \
    psql --no-psqlrc --set=ON_ERROR_STOP=1 \
    --command='SELECT 1' >/dev/null 2>&1; then
    echo "frontend operating-system user can authenticate as postgres" >&2
    return 1
  fi

  if docker exec --user autogpt_frontend "${CONTAINER_NAME}" \
    /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    /app/autogpt_platform/backend/.venv/bin/python \
    /opt/autogpt/single-container/probe.py redis --port 17000 \
    >/dev/null 2>&1; then
    echo "frontend operating-system user can access Valkey without authentication" >&2
    return 1
  fi

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

  for process_name in rest executor copilot-executor; do
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
    --workdir /app/autogpt_platform/backend \
    "${CONTAINER_NAME}" \
    /app/autogpt_platform/backend/.venv/bin/python - <<'PY'
import os
import subprocess
from pathlib import Path

from backend.integrations.codex.runtime import (
    CODEX_RUNTIME_VERSION,
    assert_pinned_versions,
    build_runtime_config,
)
from backend.integrations.codex.temporary_home import TemporaryCodexHome

root = Path(os.environ["CODEX_TEMP_ROOT"])
assert root == Path("/dev/shm/autogpt-codex")
assert_pinned_versions()
before = set(root.iterdir())
with TemporaryCodexHome.create(root) as home:
    config = build_runtime_config(home)
    launch_args = config.launch_args_override
    assert launch_args is not None
    completed = subprocess.run(
        [*launch_args[:3], "--version"],
        env=config.env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert CODEX_RUNTIME_VERSION in output, output
    home_path = home.path

assert not home_path.exists()
assert set(root.iterdir()) == before
print("codex-sdk-runtime-ok")
PY
}

assert_request_tokens_absent_from_logs() {
  local sentinel=AUTOGPT_LOG_SENTINEL_6f2b3cb87e9a
  local websocket_key=dGhlIHNhbXBsZSBub25jZQ== # pragma: allowlist secret # gitleaks:allow
  local websocket_status

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
  # Do not use grep -q here: with pipefail an early grep exit can SIGPIPE
  # `docker logs` and accidentally turn a positive match into a false result.
  if docker logs "${CONTAINER_NAME}" 2>&1 | grep -F "${sentinel}" >/dev/null; then
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
  docker exec "${CONTAINER_NAME}" \
    /usr/bin/test -f \
    /opt/prisma-python/binaries/node_modules/prisma/build/index.js
  if docker logs "${CONTAINER_NAME}" 2>&1 | \
    grep -F "Installing Prisma CLI" >/dev/null; then
    echo "Prisma CLI was installed during container startup" >&2
    return 1
  fi
}

assert_falkordb_binary_contract() {
  local linkage
  local listeners
  linkage="$(
    docker exec "${CONTAINER_NAME}" /bin/bash -Eeuo pipefail -c '
      ldd /opt/falkordb/redis-server
      ldd /opt/falkordb/falkordb.so
    '
  )"
  if grep -F "not found" <<<"${linkage}" >/dev/null; then
    echo "FalkorDB has an unresolved shared-library dependency" >&2
    return 1
  fi

  listeners="$(docker exec "${CONTAINER_NAME}" ss -lnt)"
  grep -Eq '127\.0\.0\.1:6380([[:space:]]|$)' <<<"${listeners}" || {
    echo "FalkorDB is not listening on the private loopback address" >&2
    return 1
  }
  if grep -Eq '(0\.0\.0\.0|\[::\]):6380([[:space:]]|$)' <<<"${listeners}"; then
    echo "FalkorDB is listening on a public container interface" >&2
    return 1
  fi
  if [[ -n "$(docker port "${CONTAINER_NAME}" 6380 2>/dev/null)" ]]; then
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
    current_pid="$(
      docker exec "${CONTAINER_NAME}" \
        supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
        pid falkordb 2>/dev/null || true
    )"
    if [[ "${current_pid}" =~ ^[0-9]+$ ]] &&
      [[ "${current_pid}" != "${previous_pid}" ]] &&
      appliance_is_healthy_and_armed; then
      return 0
    fi
    sleep 5
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
  docker rm --force --volumes "${NEGATIVE_CONTAINER_NAME}" >/dev/null 2>&1 || true
  ((status != 0)) || {
    echo "image accepted unsupported email verification" >&2
    return 1
  }
  ((status != 124 && status != 137)) || {
    echo "email-verification rejection probe timed out" >&2
    return 1
  }
  grep -Fq \
    "email verification is not supported by the single-container distribution" \
    <<<"${output}" || {
    echo "unsupported email verification failed without an actionable error" >&2
    return 1
  }
}

# Phase one deliberately supplies no environment, port, volume, entrypoint, or
# command override. This is the CI proof for literal `docker run IMAGE`.
assert_unsupported_email_verification_rejected
docker run --detach \
  --platform "${SMOKE_PLATFORM}" \
  --name "${CONTAINER_NAME}" \
  "${SMOKE_IMAGE}" >/dev/null
discover_data_volume
wait_for_healthy
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

docker stop --timeout 360 "${CONTAINER_NAME}" >/dev/null
[[ "$(docker inspect --format '{{.State.ExitCode}}' "${CONTAINER_NAME}")" == 0 ]] || {
  echo "container did not exit cleanly after docker stop" >&2
  exit 1
}
docker rm "${CONTAINER_NAME}" >/dev/null

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

[[ "$(runtime_config_hash)" == "${first_hash}" ]] || {
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

falkordb_pid="$(
  docker exec "${CONTAINER_NAME}" \
    supervisorctl -c /opt/autogpt/single-container/supervisor/supervisord.conf \
    pid falkordb
)"
[[ "${falkordb_pid}" =~ ^[0-9]+$ ]]
falkordb_restart_count="$(
  docker inspect --format '{{.RestartCount}}' "${CONTAINER_NAME}"
)"
docker exec "${CONTAINER_NAME}" kill -TERM "${falkordb_pid}"
wait_for_falkordb_restart "${falkordb_pid}"
[[ "$(docker inspect --format '{{.RestartCount}}' "${CONTAINER_NAME}")" == \
  "${falkordb_restart_count}" ]] || {
  echo "FalkorDB recovery unexpectedly restarted the whole container" >&2
  exit 1
}
assert_memory_contract verify
wait_for_healthy

restart_count="$(docker inspect --format '{{.RestartCount}}' "${CONTAINER_NAME}")"
docker exec "${CONTAINER_NAME}" \
  supervisorctl \
  -c /opt/autogpt/single-container/supervisor/supervisord.conf \
  stop nginx >/dev/null
wait_for_automatic_restart "$((restart_count + 1))"
[[ "$(runtime_config_hash)" == "${first_hash}" ]] || {
  echo "runtime config changed across automatic Docker restart" >&2
  exit 1
}
assert_runtime_config_mode
assert_frontend_database_isolation
assert_redirect "${PUBLIC_URL}/copilot" --resolve localhost:3300:127.0.0.1
assert_falkordb_binary_contract
assert_memory_contract cleanup

docker stop --timeout 360 "${CONTAINER_NAME}" >/dev/null
[[ "$(docker inspect --format '{{.State.ExitCode}}' "${CONTAINER_NAME}")" == 0 ]] || {
  echo "container did not exit cleanly after the restart test" >&2
  exit 1
}
docker rm "${CONTAINER_NAME}" >/dev/null

echo "single-container smoke test passed for ${SMOKE_PLATFORM}"
