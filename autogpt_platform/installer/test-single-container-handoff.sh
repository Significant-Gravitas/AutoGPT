#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT

mkdir -p \
  "${test_root}/.git" \
  "${test_root}/fake-bin" \
  "${test_root}/autogpt_platform/installer" \
  "${test_root}/autogpt_platform/single-container"

cp "${repo_root}/autogpt_platform/installer/setup-autogpt.sh" \
  "${test_root}/autogpt_platform/installer/setup-autogpt.sh"
cp "${repo_root}/autogpt_platform/single-container/.env.example" \
  "${test_root}/autogpt_platform/single-container/.env.example"
cp "${repo_root}/autogpt_platform/docker-compose.single-container.yml" \
  "${test_root}/autogpt_platform/docker-compose.single-container.yml"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'if [ "${1:-}" = info ]; then exit 0; fi' \
  'printf "fake-docker %s\\n" "$*"' \
  > "${test_root}/fake-bin/docker"
chmod +x "${test_root}/fake-bin/docker"

(
  cd "${test_root}"
  PATH="${test_root}/fake-bin:${PATH}" \
    bash autogpt_platform/installer/setup-autogpt.sh --single-container
)

env_file="${test_root}/autogpt_platform/single-container/.env"
log_file="${test_root}/autogpt_platform/logs/docker_setup.log"
test -f "${env_file}"
grep -Fqx \
  'fake-docker compose --env-file single-container/.env -f docker-compose.single-container.yml up --build --detach --wait --wait-timeout 900' \
  "${log_file}"

permissions="$(stat -c '%a' "${env_file}" 2>/dev/null || stat -f '%Lp' "${env_file}")"
test "${permissions}" = 600

bash "${repo_root}/autogpt_platform/installer/install.sh" --help \
  | grep -Fq -- '--preflight-only'
bash "${repo_root}/autogpt_platform/installer/setup-autogpt.sh" --help \
  | grep -Fq -- '--single-container'
