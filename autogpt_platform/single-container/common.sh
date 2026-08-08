#!/usr/bin/env bash

set -Eeuo pipefail

readonly AUTOGPT_RUNTIME_DIR="${AUTOGPT_RUNTIME_DIR:-/run/autogpt}"
readonly AUTOGPT_READY_FILE="${AUTOGPT_READY_FILE:-${AUTOGPT_RUNTIME_DIR}/ready}"
readonly AUTOGPT_RUNTIME_ENV="${AUTOGPT_RUNTIME_ENV:-/data/config/runtime.env}"
readonly AUTOGPT_ASSET_DIR="${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}"
readonly AUTOGPT_BACKEND_DIR="${AUTOGPT_BACKEND_DIR:-/app/autogpt_platform/backend}"
readonly AUTOGPT_FRONTEND_DIR="${AUTOGPT_FRONTEND_DIR:-/app/frontend}"

log() {
  printf '%s [single-container] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

fatal() {
  log "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fatal "required command is missing: $1"
}

validate_legacy_auth() {
  case "${AUTOGPT_ENABLE_LEGACY_AUTH:-false}" in
    true)
      if [[ -n "${JWT_VERIFY_KEY:-}" && -n "${SUPABASE_JWT_SECRET:-}" && \
        "${JWT_VERIFY_KEY}" != "${SUPABASE_JWT_SECRET}" ]]; then
        fatal "JWT_VERIFY_KEY and SUPABASE_JWT_SECRET must match during legacy auth migration"
      fi
      local legacy_secret="${JWT_VERIFY_KEY:-${SUPABASE_JWT_SECRET:-}}"
      ((${#legacy_secret} >= 32)) || \
        fatal "legacy auth requires a shared secret of at least 32 characters"
      JWT_VERIFY_KEY="${legacy_secret}"
      SUPABASE_JWT_SECRET="${legacy_secret}"
      export JWT_VERIFY_KEY SUPABASE_JWT_SECRET
      log "legacy symmetric JWT verification is explicitly enabled"
      ;;
    false | "")
      if [[ -n "${JWT_VERIFY_KEY:-}" || -n "${SUPABASE_JWT_SECRET:-}" ]]; then
        fatal "legacy JWT secrets were supplied; remove them or explicitly set AUTOGPT_ENABLE_LEGACY_AUTH=true"
      fi
      JWT_VERIFY_KEY=''
      SUPABASE_JWT_SECRET=''
      export JWT_VERIFY_KEY SUPABASE_JWT_SECRET
      ;;
    *)
      fatal "AUTOGPT_ENABLE_LEGACY_AUTH must be true or false"
      ;;
  esac
}

load_runtime_config() {
  [[ -f "${AUTOGPT_RUNTIME_ENV}" ]] || fatal "missing runtime config: ${AUTOGPT_RUNTIME_ENV}"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    local name="${line%%=*}"
    local value="${line#*=}"
    [[ "${name}" != "${line}" ]] || fatal "malformed runtime config entry"
    case "${name}" in
      AUTOGPT_RUNTIME_CONFIG_VERSION | POSTGRES_PASSWORD | RABBITMQ_DEFAULT_USER | RABBITMQ_DEFAULT_PASS | BETTER_AUTH_SECRET | ENCRYPTION_KEY | UNSUBSCRIBE_SECRET_KEY | GRAPHITI_FALKORDB_PASSWORD | VAPID_PRIVATE_KEY | VAPID_PUBLIC_KEY)
        export "${name}=${value}"
        ;;
      *)
        fatal "unknown key in runtime config: ${name}"
        ;;
    esac
  done <"${AUTOGPT_RUNTIME_ENV}"
}

wait_for_ready_file() {
  local timeout="${AUTOGPT_STARTUP_TIMEOUT:-600}"
  local elapsed=0
  while [[ ! -f "${AUTOGPT_READY_FILE}" ]]; do
    ((elapsed < timeout)) || fatal "startup did not complete within ${timeout}s"
    sleep 1
    ((elapsed += 1))
  done
}

run_as() {
  local user="$1"
  shift
  runuser --user "${user}" -- "$@"
}

run_rabbitmq_cli() {
  run_as rabbitmq /usr/bin/env -i \
    PATH=/opt/rabbitmq/sbin:/opt/erlang/bin:/opt/openssl/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/data/rabbitmq \
    LANG=C.UTF-8 \
    RABBITMQ_NODENAME=rabbit@localhost \
    ERL_EPMD_ADDRESS=127.0.0.1 \
    "$@"
}
