#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

readonly SUPERVISOR_PID_FILE="${AUTOGPT_RUNTIME_DIR}/supervisord.pid"
readonly WATCHDOG_ARMED_FILE="${AUTOGPT_RUNTIME_DIR}/watchdog-armed"
readonly FAILURE_LIMIT=3
readonly INITIAL_CHECK_INTERVAL=1
readonly CHECK_INTERVAL=30
readonly CHECK_TIMEOUT=60
readonly INITIAL_HEALTH_TIMEOUT=600

CHECK_TIMER_PID=
FORCED_CHECK_PENDING=false
CHECK_TRIGGER=scheduled
WATCHDOG_OUTPUT=

main() {
  local failures=0
  rm -f "${WATCHDOG_ARMED_FILE}"
  WATCHDOG_OUTPUT="$(mktemp "${AUTOGPT_RUNTIME_DIR}/watchdog-health.XXXXXX")"
  trap 'rm -f "${WATCHDOG_OUTPUT}" "${WATCHDOG_ARMED_FILE}"' EXIT
  trap queue_forced_check USR1

  wait_for_ready_file
  wait_for_initial_health "${WATCHDOG_OUTPUT}"
  while true; do
    wait_for_next_check
    if run_healthcheck "${WATCHDOG_OUTPUT}"; then
      failures=0
      if [[ "${CHECK_TRIGGER}" == forced ]]; then
        log "watchdog health check passed trigger=forced"
      fi
      continue
    fi

    ((failures += 1))
    log "watchdog health failure ${failures}/${FAILURE_LIMIT} trigger=${CHECK_TRIGGER}" >&2
    sed -n '1,120p' "${WATCHDOG_OUTPUT}" >&2
    if ((failures >= FAILURE_LIMIT)); then
      stop_appliance \
        "watchdog stopping the unhealthy appliance; a Docker restart policy is required to restart it automatically"
    fi
  done
}

queue_forced_check() {
  # USR1 advances one cadence slot; the normal healthcheck and failure limit
  # below remain the only path that can stop the appliance.
  FORCED_CHECK_PENDING=true
  if [[ -n "${CHECK_TIMER_PID}" ]]; then
    if ! kill -TERM "${CHECK_TIMER_PID}" 2>/dev/null; then
      :
    fi
  fi
}

wait_for_next_check() {
  local timer_pid
  local timer_status
  CHECK_TRIGGER=scheduled

  if [[ "${FORCED_CHECK_PENDING}" == true ]]; then
    FORCED_CHECK_PENDING=false
    CHECK_TRIGGER=forced
    return 0
  fi

  sleep "${CHECK_INTERVAL}" &
  timer_pid=$!
  CHECK_TIMER_PID="${timer_pid}"
  if [[ "${FORCED_CHECK_PENDING}" == true ]]; then
    if ! kill -TERM "${timer_pid}" 2>/dev/null; then
      :
    fi
  fi
  if wait "${timer_pid}"; then
    timer_status=0
  else
    timer_status=$?
  fi
  CHECK_TIMER_PID=

  if [[ "${FORCED_CHECK_PENDING}" == true ]]; then
    FORCED_CHECK_PENDING=false
    CHECK_TRIGGER=forced
    if kill -0 "${timer_pid}" 2>/dev/null; then
      if ! kill -TERM "${timer_pid}" 2>/dev/null; then
        :
      fi
    fi
    # Reap the timer if the signal interrupted `wait` before TERM completed.
    if ! wait "${timer_pid}" 2>/dev/null; then
      :
    fi
    return 0
  fi

  ((timer_status == 0)) || fatal "watchdog check timer failed with status ${timer_status}"
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
      log "watchdog armed after initial healthy state; automatic fatal recovery requires a Docker restart policy"
      return 0
    fi
    log "watchdog waiting for initial healthy state" >&2
    sed -n '1,120p' "${output}" >&2
    sleep "${INITIAL_CHECK_INTERVAL}"
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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
