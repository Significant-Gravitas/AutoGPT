#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLATFORM_DIR="${ROOT_DIR}/autogpt_platform"

export NVM_DIR="${NVM_DIR:-/opt/nvm}"
if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
  # shellcheck disable=SC1091
  . "${NVM_DIR}/nvm.sh"
  nvm use 24 >/dev/null
fi

export PATH="${NVM_DIR}/versions/node/v24.19.0/bin:${HOME}/.local/bin:${PATH}"

cd "${PLATFORM_DIR}"

make init-env

cd backend
poetry install --with dev --no-interaction
poetry run prisma generate
poetry run gen-prisma-stub

cd ../frontend
pnpm install --frozen-lockfile
