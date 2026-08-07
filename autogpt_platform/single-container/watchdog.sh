#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

readonly SUPERVISOR_PID_FILE="${AUTOGPT_RUNTIME_DIR}/supervisord.pid"
readonly WATCHDOG_ARMED_FILE="${AUTOGPT_RUNTIME_DIR}/watchdog-armed"
readonly FAILURE_LIMIT=3
readonly CHECK_INTERVAL=30
readonly CHECK_TIMEOUT=60
readonly INITIAL_HEALTH_TIMEOUT=600

main() {
  local failures=0
  local output
  rm -f "${WATCHDOG_ARMED_FILE}"
  output="$(mktemp "${AUTOGPT_RUNTIME_DIR}/watchdog-health.XXXXXX")"
  trap 'rm -f "${output}" "${WATCHDOG_ARMED_FILE}"' EXIT

  wait_for_ready_file
  wait_for_initial_health "${output}"
  while true; do
    sleep "${CHECK_INTERVAL}"
    if run_healthcheck "${output}"; then
      failures=0
      continue
    fi

    ((failures += 1))
    log "watchdog health failure ${failures}/${FAILURE_LIMIT}" >&2
    sed -n '1,120p' "${output}" >&2
    if ((failures >= FAILURE_LIMIT)); then
      stop_appliance "watchdog stopping the unhealthy appliance for Docker restart"
    fi
  done
}

run_healthcheck() {
  local output="$1"
  timeout --signal=TERM --kill-after=5s "${CHECK_TIMEOUT}" \
    /usr/local/bin/autogpt-healthcheck >"${output}" 2>&1
}

wait_for_initial_health() {
  local output="$1"
  local deadline=$((SECONDS + INITIAL_HEALTH_TIMEOUT))
  while ((SECONDS < deadline)); do
    if run_healthcheck "${output}"; then
      install -m 0600 /dev/null "${WATCHDOG_ARMED_FILE}"
      log "watchdog armed after initial healthy state"
      return 0
    fi
    log "watchdog waiting for initial healthy state" >&2
    sed -n '1,120p' "${output}" >&2
    sleep "${CHECK_INTERVAL}"
  done
  stop_appliance \
    "watchdog did not observe a healthy appliance within ${INITIAL_HEALTH_TIMEOUT}s"
}

stop_appliance() {
  local reason="$1"
  [[ -s "${SUPERVISOR_PID_FILE}" ]] || fatal "supervisor PID file is missing"
  log "${reason}" >&2
  kill -TERM "$(<"${SUPERVISOR_PID_FILE}")"
  exit 0
}

main "$@"
