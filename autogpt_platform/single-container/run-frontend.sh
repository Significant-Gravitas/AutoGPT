#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

readonly FRONTEND_DATABASE_URL='postgresql:///postgres?host=%2Frun%2Fpostgresql&user=autogpt_frontend'

readonly -a REQUIRED_FRONTEND_ENV=(
  AGPT_SERVER_URL
  AGPT_WS_SERVER_URL
  AUTH_ALLOW_NEW_ACCOUNTS
  AUTH_DB_SCHEMA
  AUTH_REQUIRE_EMAIL_VERIFICATION
  BETTER_AUTH_INTERNAL_URL
  BETTER_AUTH_SECRET
  BETTER_AUTH_URL
)

readonly -a OPTIONAL_FRONTEND_ENV=(
  AUTH_CALLBACK_URL
  AUTH_DISCORD_CLIENT_ID
  AUTH_DISCORD_CLIENT_SECRET
  AUTH_GITHUB_CLIENT_ID
  AUTH_GITHUB_CLIENT_SECRET
  AUTH_GOOGLE_CLIENT_ID
  AUTH_GOOGLE_CLIENT_SECRET
  AUTH_SIGNUP_ALLOWLIST
  OPENAI_API_BASE_URL
  OPENAI_API_KEY
  SUPABASE_BRIDGE_MAX_TOKEN_AGE_DAYS
  SUPABASE_JWT_SECRET
  TRANSCRIPTION_API_BASE_URL
  TRANSCRIPTION_API_KEY
  TRANSCRIPTION_MODEL
)

declare -a frontend_env=()

build_frontend_environment() {
  local name
  frontend_env=(
    PATH=/usr/local/bin:/usr/bin:/bin
    HOME=/data/frontend-home
    XDG_CACHE_HOME=/data/cache/next
    USER=autogpt_frontend
    LOGNAME=autogpt_frontend
    DATABASE_URL="${FRONTEND_DATABASE_URL}"
    LANG=C.UTF-8
    LC_ALL=C.UTF-8
    NODE_ENV=production
    NEXT_TELEMETRY_DISABLED=1
    PORT=3001
    HOSTNAME=127.0.0.1
  )

  for name in "${REQUIRED_FRONTEND_ENV[@]}"; do
    [[ -n "${!name:-}" ]] || fatal "required frontend setting is missing: ${name}"
    frontend_env+=("${name}=${!name}")
  done
  for name in "${OPTIONAL_FRONTEND_ENV[@]}"; do
    if [[ -v "${name}" ]]; then
      frontend_env+=("${name}=${!name}")
    fi
  done
}

main() {
  [[ "$(id -u)" -eq 0 ]] || fatal "frontend launcher must start as root"
  build_frontend_environment

  wait_for_ready_file
  (($# > 0)) || set -- node /app/frontend/server.js
  log "starting next"
  exec /usr/bin/env -i "${frontend_env[@]}" \
    /usr/bin/setpriv \
    --reuid=autogpt_frontend \
    --regid=autogpt_frontend \
    --clear-groups \
    --nnp \
    --bounding-set=-all \
    --inh-caps=-all \
    --ambient-caps=-all \
    -- "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
