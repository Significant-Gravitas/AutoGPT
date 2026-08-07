#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
test_root="$(mktemp -d)"
trap 'rm -rf "${test_root}"' EXIT

mkdir -p \
  "${test_root}/.git" \
  "${test_root}/fake-bin" \
  "${test_root}/preflight-bin" \
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

install_help="$(bash "${repo_root}/autogpt_platform/installer/install.sh" --help)"
setup_help="$(bash "${repo_root}/autogpt_platform/installer/setup-autogpt.sh" --help)"
grep -Fq -- '--preflight-only' <<< "${install_help}"
grep -Fq -- '--single-container' <<< "${setup_help}"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  '[ -z "${FAKE_CURL_MARKER:-}" ] || : > "${FAKE_CURL_MARKER}"' \
  'printf "%s\\n" "{\"tag_name\": \"autogpt-installer-test-v1.2.3\"}"' \
  > "${test_root}/fake-bin/curl"
chmod +x "${test_root}/fake-bin/curl"

resolve_output="$(
  PATH="${test_root}/fake-bin:${PATH}" \
    bash "${repo_root}/autogpt_platform/installer/install.sh" --resolve-only
)"
grep -Fq 'Selected version -> tag: autogpt-installer-test-v1.2.3' \
  <<< "${resolve_output}"

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'case "${1:-}" in' \
  '  -s) printf "%s\\n" "${FAKE_UNAME_S:-Linux}" ;;' \
  '  -m) printf "%s\\n" "x86_64" ;;' \
  'esac' \
  > "${test_root}/preflight-bin/uname"
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'if [ "${2:-}" = /proc/meminfo ]; then' \
  '  printf "%s\\n" "${FAKE_RAM_KB:?}"' \
  'else' \
  '  cat >/dev/null' \
  '  printf "%s\\n" "${FAKE_FREE_GB:?}"' \
  'fi' \
  > "${test_root}/preflight-bin/awk"
printf '%s\n' '#!/usr/bin/env bash' 'printf "fake df\\n"' \
  > "${test_root}/preflight-bin/df"
printf '%s\n' '#!/usr/bin/env bash' 'printf "0\\n"' \
  > "${test_root}/preflight-bin/id"
chmod +x "${test_root}/preflight-bin/"*

preflight_path="${test_root}/preflight-bin:${test_root}/fake-bin:${PATH}"
preflight_output="$(
  PATH="${preflight_path}" FAKE_RAM_KB=8388608 FAKE_FREE_GB=25 \
    bash "${repo_root}/autogpt_platform/installer/install.sh" --preflight-only
)"
grep -Fq '[ OK ] 25 GB free' <<< "${preflight_output}"
grep -Fq '[ OK ] 8 GB RAM' <<< "${preflight_output}"

curl_marker="${test_root}/curl-was-called"
if PATH="${preflight_path}" \
  FAKE_RAM_KB=8388608 FAKE_FREE_GB=24 FAKE_CURL_MARKER="${curl_marker}" \
  bash "${repo_root}/autogpt_platform/installer/install.sh" --preflight-only \
  > "${test_root}/preflight-failure.log" 2>&1; then
  echo "24 GB preflight unexpectedly passed" >&2
  exit 1
fi
grep -Fq '[FAIL] Only 24 GB free' "${test_root}/preflight-failure.log"
test ! -e "${curl_marker}"

if PATH="${preflight_path}" \
  FAKE_UNAME_S=Plan9 FAKE_RAM_KB=8388608 FAKE_FREE_GB=25 \
  FAKE_CURL_MARKER="${curl_marker}" \
  bash "${repo_root}/autogpt_platform/installer/install.sh" --preflight-only \
  > "${test_root}/unsupported-os.log" 2>&1; then
  echo "unsupported OS preflight unexpectedly passed" >&2
  exit 1
fi
grep -Fq 'Unsupported OS: Plan9' "${test_root}/unsupported-os.log"
test ! -e "${curl_marker}"
