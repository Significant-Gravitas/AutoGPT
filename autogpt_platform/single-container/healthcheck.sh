#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

readonly PROBE=("${AUTOGPT_BACKEND_DIR}/.venv/bin/python" "${AUTOGPT_ASSET_DIR}/probe.py")
readonly SUPERVISOR_CONFIG="${AUTOGPT_ASSET_DIR}/supervisor/supervisord.conf"
readonly POSTGRES_BINDIR="${POSTGRES_BINDIR:-/usr/lib/postgresql/15/bin}"

main() {
  [[ -f "${AUTOGPT_READY_FILE}" ]] || fatal "bootstrap readiness marker is absent"
  load_runtime_config
  check_supervised_processes
  check_infrastructure
  check_application_services
}

check_supervised_processes() {
  local programs=(
    fatal-exit postgres valkey-0 valkey-1 valkey-2 rabbitmq falkordb
    database-manager scheduler batch-executor notification executor
    copilot-executor copilot-bot platform-linking-manager websocket rest next nginx
    watchdog
  )
  local program
  local statuses
  # `supervisorctl status` exits non-zero when any program is not RUNNING. The
  # bootstrap program is intentionally one-shot, so validate only the required
  # long-running programs below instead of treating its expected EXITED state as
  # a failed Supervisor connection.
  statuses="$(supervisorctl -c "${SUPERVISOR_CONFIG}" status 2>/dev/null || true)"
  [[ -n "${statuses}" ]] || fatal "could not read supervisor process status"
  for program in "${programs[@]}"; do
    grep -Eq "^${program}[[:space:]]+RUNNING([[:space:]]|$)" <<<"${statuses}" || \
      fatal "supervisor process is not running: ${program}"
  done
}

check_infrastructure() {
  "${POSTGRES_BINDIR}/pg_isready" -q -h 127.0.0.1 -p 5432 -U postgres
  "${PROBE[@]}" redis --port 17000 --cluster \
    --password-env REDIS_PASSWORD
  "${PROBE[@]}" amqp --port 5672 \
    --username-env RABBITMQ_DEFAULT_USER \
    --password-env RABBITMQ_DEFAULT_PASS
  "${PROBE[@]}" redis --port 6380 \
    --password-env GRAPHITI_FALKORDB_PASSWORD
}

check_application_services() {
  local urls=(
    "http://127.0.0.1:${AUTOGPT_DATABASE_API_PORT}${AUTOGPT_INTERNAL_HEALTH_PATH}"
    "http://127.0.0.1:${AUTOGPT_EXECUTION_SCHEDULER_PORT}${AUTOGPT_INTERNAL_HEALTH_PATH}"
    "http://127.0.0.1:${AUTOGPT_BATCH_EXECUTOR_PORT}${AUTOGPT_INTERNAL_HEALTH_PATH}"
    "http://127.0.0.1:${AUTOGPT_NOTIFICATION_SERVICE_PORT}${AUTOGPT_INTERNAL_HEALTH_PATH}"
    "http://127.0.0.1:${AUTOGPT_EXECUTION_MANAGER_PORT}${AUTOGPT_INTERNAL_HEALTH_PATH}"
    "http://127.0.0.1:${AUTOGPT_COPILOT_EXECUTOR_PORT}${AUTOGPT_INTERNAL_HEALTH_PATH}"
    "http://127.0.0.1:${AUTOGPT_WEBSOCKET_PORT}/health"
    "http://127.0.0.1:${AUTOGPT_AGENT_API_PORT}/health"
    http://127.0.0.1:3001/
    http://127.0.0.1:3000/healthz
  )
  if [[ "${AUTOGPT_ENABLE_BOT_SERVICES:-false}" == true ]]; then
    urls+=(
      "http://127.0.0.1:${AUTOGPT_COPILOT_CHAT_BRIDGE_PORT}${AUTOGPT_INTERNAL_HEALTH_PATH}"
      "http://127.0.0.1:${AUTOGPT_PLATFORM_LINKING_SERVICE_PORT}${AUTOGPT_INTERNAL_HEALTH_PATH}"
    )
  fi
  "${PROBE[@]}" http --timeout 10 "${urls[@]}" || \
    fatal "one or more application health probes failed"
}

main "$@"
