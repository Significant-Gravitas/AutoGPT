#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

toggle_name="${1:-}"
name="${2:-optional application}"
shift 2 || true
(($# > 0)) || fatal "no command supplied for ${name}"

case "${!toggle_name:-}" in
  true)
    exec "${AUTOGPT_ASSET_DIR}/run-app.sh" "${name}" "$@"
    ;;
  false)
    wait_for_ready_file
    exec "${AUTOGPT_ASSET_DIR}/disabled-service.sh" "${name}"
    ;;
  *)
    fatal "${toggle_name} must be true or false"
    ;;
esac
