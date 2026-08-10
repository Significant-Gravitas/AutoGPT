#!/usr/bin/env bash

set -Eeuo pipefail

# shellcheck source=common.sh
source "${AUTOGPT_ASSET_DIR:-/opt/autogpt/single-container}/common.sh"

service="${1:-}"

case "${service}" in
  postgres)
    postgres_bindir="${POSTGRES_BINDIR:-/usr/lib/postgresql/15/bin}"
    exec "${postgres_bindir}/postgres" -D "${PGDATA:-/data/postgres}"
    ;;
  valkey-0 | valkey-1 | valkey-2)
    index="${service##*-}"
    port="$((17000 + index))"
    exec valkey-server "${AUTOGPT_RUNTIME_DIR}/valkey/${port}.conf"
    ;;
  rabbitmq)
    exec /opt/rabbitmq/sbin/rabbitmq-server
    ;;
  falkordb)
    exec /opt/falkordb/redis-server "${AUTOGPT_RUNTIME_DIR}/falkordb.conf"
    ;;
  clamd)
    if [[ "${AUTOGPT_ENABLE_CLAMAV:-true}" == false ]]; then
      exec "${AUTOGPT_ASSET_DIR}/disabled-service.sh" clamd
    fi
    exec clamd --foreground=true --config-file="${AUTOGPT_ASSET_DIR}/clamav/clamd.conf"
    ;;
  freshclam)
    if [[ "${AUTOGPT_ENABLE_CLAMAV:-true}" == false ]]; then
      exec "${AUTOGPT_ASSET_DIR}/disabled-service.sh" freshclam
    fi
    exec freshclam --daemon --foreground \
      --config-file="${AUTOGPT_ASSET_DIR}/clamav/freshclam.conf"
    ;;
  *)
    fatal "unknown bundled service: ${service:-<empty>}"
    ;;
esac
