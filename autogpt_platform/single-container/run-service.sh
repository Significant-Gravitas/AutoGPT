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
    exec valkey-server \
      --bind 127.0.0.1 \
      --protected-mode yes \
      --port "${port}" \
      --dir "/data/valkey/${port}" \
      --appendonly yes \
      --cluster-enabled yes \
      --cluster-config-file nodes.conf \
      --cluster-node-timeout 5000 \
      --cluster-require-full-coverage no \
      --cluster-announce-ip 127.0.0.1 \
      --cluster-announce-port "${port}" \
      --cluster-announce-bus-port "$((port + 10000))"
    ;;
  rabbitmq)
    exec /opt/rabbitmq/sbin/rabbitmq-server
    ;;
  falkordb)
    if [[ "${AUTOGPT_ENABLE_FALKORDB:-true}" == false ]]; then
      exec "${AUTOGPT_ASSET_DIR}/disabled-service.sh" falkordb
    fi
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
