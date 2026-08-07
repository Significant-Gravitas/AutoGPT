#!/usr/bin/env bash

set -Eeuo pipefail

: "${SMOKE_IMAGE:?SMOKE_IMAGE is required}"
: "${SMOKE_PLATFORM:?SMOKE_PLATFORM is required}"

readonly PUBLIC_URL=http://localhost:3300
readonly TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-2700}"
readonly SAFE_PLATFORM="${SMOKE_PLATFORM//\//-}"
readonly RUN_TOKEN="${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}-${SAFE_PLATFORM}-${RANDOM}"
readonly RUN_CONTAINER_NAME="autogpt-single-smoke-${RUN_TOKEN}"
HEADERS_FILE="$(mktemp)"
readonly HEADERS_FILE

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
  trap - EXIT INT TERM
  if ((result != 0)); then
    diagnostics
  fi
  if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    docker stop --timeout 360 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm --force --volumes "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${DATA_VOLUME}" ]] && docker volume inspect "${DATA_VOLUME}" >/dev/null 2>&1; then
    docker volume rm "${DATA_VOLUME}" >/dev/null 2>&1 || true
  fi
  rm -f "${HEADERS_FILE}"
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
  local mode
  mode="$(docker exec "${CONTAINER_NAME}" stat -c '%a' /data/config/runtime.env)"
  [[ "${mode}" == 600 ]] || {
    echo "runtime config mode is ${mode}, expected 600" >&2
    return 1
  }
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
  set +e
  output="$(
    docker run --rm \
      --platform "${SMOKE_PLATFORM}" \
      --env AUTH_REQUIRE_EMAIL_VERIFICATION=true \
      "${SMOKE_IMAGE}" 2>&1
  )"
  status=$?
  set -e
  ((status != 0)) || {
    echo "image accepted unsupported email verification" >&2
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
assert_falkordb_binary_contract
assert_memory_contract seed

first_hash="$(runtime_config_hash)"
[[ "${first_hash}" =~ ^[0-9a-f]{64}$ ]] || {
  echo "runtime config checksum is invalid" >&2
  exit 1
}
assert_runtime_config_mode

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
assert_pinned_topology_environment
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
