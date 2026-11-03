#!/usr/bin/env bash
set -Eeuo pipefail
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"
name="${1:-application}"
shift || true
(($# > 0)) || fatal "no command supplied for ${name}"
export HOME="${AUTOGPT_HOME:-/data/home}"
export XDG_CACHE_HOME="${AUTOGPT_CACHE_DIR:-/data/cache/backend}"
case "$1" in
  rest|db|ws|scheduler|notification|executor|copilot-executor|batch-executor)
    log "preloading ${name} before bootstrap readiness"
    exec "${AUTOGPT_BACKEND_DIR}/.venv/bin/python" \
      "${AUTOGPT_ASSET_DIR}/run-python-app-preload.py" "$@"
    ;;
  *)
    wait_for_ready_file
    log "starting ${name}"
    exec "$@"
    ;;
esac
