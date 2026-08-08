#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLATFORM_DIR="${ROOT_DIR}/autogpt_platform"
COMPOSE=(docker compose -f docker-compose.yml -f ../.cursor/docker-compose.cloud.yml)

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
  # shellcheck disable=SC1091
  . "${NVM_DIR}/nvm.sh"
  nvm use 24 >/dev/null
fi

export PATH="${NVM_DIR}/versions/node/v24.19.0/bin:${HOME}/.local/bin:${PATH}"

sudo sysctl -w vm.overcommit_memory=1 >/dev/null 2>&1 || true

if ! docker info >/dev/null 2>&1; then
  if ! pgrep -x dockerd >/dev/null 2>&1; then
    sudo dockerd --iptables=false >/tmp/dockerd.log 2>&1 &
    for _ in $(seq 1 30); do
      if docker info >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
  fi
  if [[ ! -S /var/run/docker.sock ]] || ! docker info >/dev/null 2>&1; then
    echo "Docker daemon failed to start" >&2
    tail -50 /tmp/dockerd.log >&2 || true
    exit 1
  fi
  sudo chmod 666 /var/run/docker.sock 2>/dev/null || true
fi

cd "${PLATFORM_DIR}"

"${COMPOSE[@]}" down --remove-orphans 2>/dev/null || true
"${COMPOSE[@]}" up -d db rabbitmq clamav falkordb redis-0

for _ in $(seq 1 30); do
  if "${COMPOSE[@]}" exec -T db pg_isready -U postgres -h localhost >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

for _ in $(seq 1 30); do
  if docker run --rm --network host redis:7 redis-cli -h 127.0.0.1 -p 17000 ping >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! docker run --rm --network host redis:7 redis-cli -h 127.0.0.1 -p 17000 cluster info 2>/dev/null | grep -q 'cluster_state:ok'; then
  docker run --rm --network host redis:7 sh -c '
    NODE_ID=$(redis-cli -h 127.0.0.1 -p 17000 cluster myid)
    redis-cli -h 127.0.0.1 -p 17000 cluster addslots $(seq 0 16383)
  '
fi

make migrate
