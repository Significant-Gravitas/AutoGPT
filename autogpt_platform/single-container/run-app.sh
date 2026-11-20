#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

name="${1:-application}"
shift || true
(($# > 0)) || fatal "no command supplied for ${name}"

wait_for_ready_file
log "starting ${name}"
export HOME="${AUTOGPT_HOME:-/data/home}"
export XDG_CACHE_HOME="${AUTOGPT_CACHE_DIR:-/data/cache/backend}"
exec "$@"
