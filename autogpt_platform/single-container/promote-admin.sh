#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

readonly POSTGRES_BINDIR="${POSTGRES_BINDIR:-/usr/lib/postgresql/15/bin}"

main() {
  local email
  if (($# == 2)) && [[ "$1" == promote ]]; then
    email="$2"
  elif (($# == 1)); then
    # Retain the original direct-script form for compatibility.
    email="$1"
  else
    fatal "usage: autogpt-admin promote EMAIL"
  fi
  [[ "${email}" =~ ^[^[:space:]]+@[^[:space:]]+$ ]] || fatal "invalid email address"

  load_runtime_config
  wait_for_ready_file

  local updated
  updated="$(
    PGPASSWORD="${POSTGRES_PASSWORD}" \
      "${POSTGRES_BINDIR}/psql" \
      --host=127.0.0.1 \
      --port=5432 \
      --username=postgres \
      --dbname=postgres \
      --no-psqlrc \
      --tuples-only \
      --no-align \
      --set=ON_ERROR_STOP=1 \
      --set="target_email=${email}" <<'SQL'
WITH target AS (
  SELECT id
  FROM platform."UserAuthIdentity"
  WHERE lower(email) = lower(:'target_email')
), promoted AS (
  UPDATE platform."UserAuthIdentity"
  SET role = 'admin', "updatedAt" = now()
  WHERE id = (SELECT id FROM target LIMIT 1)
    AND (SELECT count(*) FROM target) = 1
  RETURNING id
)
SELECT count(*) FROM promoted;
SQL
  )"

  [[ "${updated}" == "1" ]] || fatal "no unique Better Auth user found for ${email}"
  log "promoted ${email} to administrator; sign out and back in to refresh the session role"
}

main "$@"
