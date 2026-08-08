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
    fatal-exit postgres valkey-0 valkey-1 valkey-2 rabbitmq falkordb clamd freshclam clamav-logs
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
  "${PROBE[@]}" redis --port 17000 --cluster
  "${PROBE[@]}" tcp --port 5672
  "${PROBE[@]}" redis --port 6380 \
    --password-env GRAPHITI_FALKORDB_PASSWORD
  if [[ "${AUTOGPT_ENABLE_CLAMAV:-true}" == true ]]; then
    "${PROBE[@]}" clam --port 3310
  fi
}

check_application_services() {
  local urls=(
    http://127.0.0.1:8005/health_check
    http://127.0.0.1:8003/health_check
    http://127.0.0.1:8011/health_check
    http://127.0.0.1:8007/health_check
    http://127.0.0.1:8002/health_check
    http://127.0.0.1:8008/health_check
    http://127.0.0.1:8001/health
    http://127.0.0.1:8006/health
    http://127.0.0.1:3001/
    http://127.0.0.1:3000/healthz
  )
  local url
  local pid
  local failed=0
  local -a pids=()
  if [[ "${AUTOGPT_ENABLE_BOT_SERVICES:-false}" == true ]]; then
    urls+=(
      http://127.0.0.1:8010/health_check
      http://127.0.0.1:8009/health_check
    )
  fi
  for url in "${urls[@]}"; do
    (
      "${PROBE[@]}" http "${url}" --timeout 10 || \
        fatal "application health probe failed: ${url}"
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  ((failed == 0)) || fatal "one or more application health probes failed"
}

main "$@"
