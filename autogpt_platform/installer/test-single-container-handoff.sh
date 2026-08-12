#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
installer="${repo_root}/autogpt_platform/installer/install.sh"
test_root="$(mktemp -d)"
test_root="$(cd "$test_root" && pwd -P)"
trap 'rm -rf "${test_root}"' EXIT

assert_contains() {
	local expected="$1" actual="$2"
	grep -Fq -- "$expected" <<<"$actual" || {
		printf 'missing expected text: %s\nactual output:\n%s\n' "$expected" "$actual" >&2
		return 1
	}
}

assert_not_contains() {
	local unexpected="$1" actual="$2"
	if grep -Fq -- "$unexpected" <<<"$actual"; then
		printf 'unexpected text: %s\nactual output:\n%s\n' "$unexpected" "$actual" >&2
		return 1
	fi
}

assert_fails_with() {
	local expected="$1"
	shift
	local output
	if output="$("$@" 2>&1)"; then
		printf 'command unexpectedly succeeded: %q ' "$@" >&2
		printf '\n' >&2
		return 1
	fi
	assert_contains "$expected" "$output"
}

file_mode() {
	if stat -c '%a' "$1" >/dev/null 2>&1; then stat -c '%a' "$1"; else stat -f '%Lp' "$1"; fi
}

default_home="${test_root}/default-home"
mkdir -m 0700 "$default_home"
default_dir_output="$(env -u XDG_CONFIG_HOME HOME="$default_home" bash "$installer" --check-dir-only)"
assert_contains "Validated install directory: ${default_home}/.config/autogpt" "$default_dir_output"
xdg_trailing_output="$(XDG_CONFIG_HOME="${default_home}/.config/" HOME="$default_home" bash "$installer" --check-dir-only)"
assert_contains "Validated install directory: ${default_home}/.config/autogpt" "$xdg_trailing_output"

bootstrap_bin="${test_root}/bootstrap-bin"
bootstrap_executed="${test_root}/bootstrap-executed"
mkdir "$bootstrap_bin"
cat >"${bootstrap_bin}/curl" <<'FAKE_CURL'
#!/bin/sh
set -eu
output=''
while [ $# -gt 0 ]; do
	if [ "$1" = -o ]; then output="$2"; shift; fi
	shift
done
printf 'partial installer\n' >"$output"
exit 22
FAKE_CURL
cat >"${bootstrap_bin}/bash" <<'FAKE_BASH'
#!/bin/sh
touch "${BOOTSTRAP_EXECUTED:?}"
exit 0
FAKE_BASH
chmod +x "${bootstrap_bin}/curl" "${bootstrap_bin}/bash"
if PATH="${bootstrap_bin}:${PATH}" BOOTSTRAP_EXECUTED="$bootstrap_executed" /bin/sh -c '
  ( installer="$(mktemp)" && trap '\''rm -f "$installer"'\'' EXIT &&
    curl --proto '\''=https'\'' --proto-redir '\''=https'\'' --tlsv1.2 -fsSL \
      -o "$installer" https://setup.agpt.co/install.sh && bash "$installer" )
'; then
	printf 'failed bootstrap download unexpectedly succeeded\n' >&2
	exit 1
fi
[ ! -e "$bootstrap_executed" ] || {
	printf 'partial bootstrap content was executed after a failed download\n' >&2
	exit 1
}

bootstrap_docs=(docs/platform/installer.md autogpt_platform/installer/install.sh)
if grep -En 'curl[^`]*\|[^`]*bash|install\.(bat|ps1)|Invoke-WebRequest' "${bootstrap_docs[@]}"; then
	printf 'unsafe or stale installer bootstrap remains in public docs\n' >&2
	exit 1
fi
grep -Fq -- "--proto '=https' --proto-redir '=https' --tlsv1.2" autogpt_platform/installer/install.sh

latest_output="$(bash "$installer" --resolve-only)"
assert_contains 'Selected release -> latest -> docker.io/significantgravitas/autogpt:latest' "$latest_output"

release_output="$(bash "$installer" --resolve-only --release autogpt-platform-beta-v1.2.3)"
assert_contains 'Selected release -> autogpt-platform-beta-v1.2.3 -> docker.io/significantgravitas/autogpt:v1.2.3' "$release_output"

short_release_output="$(bash "$installer" --resolve-only --release v2.3.4)"
assert_contains 'Selected release -> v2.3.4 -> docker.io/significantgravitas/autogpt:v2.3.4' "$short_release_output"

assert_fails_with 'Expected autogpt-platform-beta-vX.Y.Z or vX.Y.Z' bash "$installer" --resolve-only --release latest
assert_fails_with 'needs a non-empty value' bash "$installer" --resolve-only --release=
assert_fails_with 'Unknown flag: --dev' bash "$installer" --resolve-only --dev
assert_fails_with 'Unknown flag: --branch' bash "$installer" --resolve-only --branch dev
assert_fails_with 'needs a value, not another option' bash "$installer" --release --resolve-only
assert_fails_with 'needs a value, not another option' bash "$installer" --dir --resolve-only

assert_fails_with 'Refusing broad or system install directory: /' bash "$installer" --check-dir-only --dir /
assert_fails_with 'Refusing broad or system install directory' bash "$installer" --check-dir-only --dir "$HOME"
assert_fails_with "contains an unsafe '.' component" bash "$installer" --check-dir-only --dir "$HOME/."
assert_fails_with 'contains redundant path separators' bash "$installer" --check-dir-only --dir "$HOME//"

existing_dir="${test_root}/existing-install"
mkdir -m 0700 "$existing_dir"
bash "$installer" --check-dir-only --dir "$existing_dir" >/dev/null
test "$(file_mode "$existing_dir")" = 700

nonprivate_dir="${test_root}/nonprivate-install"
mkdir -m 0751 "$nonprivate_dir"
assert_fails_with 'must already be private mode 0700' bash "$installer" --check-dir-only --dir "$nonprivate_dir"

shared_dir="${test_root}/shared-install"
mkdir -m 0777 "$shared_dir"
assert_fails_with 'writable by another user or group' bash "$installer" --check-dir-only --dir "$shared_dir"

symlink_target="${test_root}/symlink-target"
symlink_dir="${test_root}/symlink-install"
mkdir "$symlink_target"
ln -s "$symlink_target" "$symlink_dir"
assert_fails_with 'symlink' bash "$installer" --check-dir-only --dir "$symlink_dir"

intermediate_target="${test_root}/intermediate-target"
intermediate_link="${test_root}/intermediate-link"
mkdir "$intermediate_target"
ln -s "$intermediate_target" "$intermediate_link"
assert_fails_with 'symlink component' bash "$installer" --check-dir-only --dir "${intermediate_link}/child"

if [ "$(uname -s)" = Darwin ]; then
	acl_dir="${test_root}/acl-install"
	mkdir -m 0700 "$acl_dir"
	chmod +a 'everyone allow read,write,execute,delete,append,readattr,writeattr,readextattr,writeextattr,readsecurity,writesecurity,chown' "$acl_dir"
	assert_fails_with 'ACL' bash "$installer" --check-dir-only --dir "$acl_dir"
	for acl_right in delete chown; do
		acl_parent="${test_root}/acl-parent-${acl_right}"
		mkdir -m 0700 "$acl_parent"
		chmod +a "everyone allow ${acl_right}" "$acl_parent"
		assert_fails_with 'unsafe writable ACL' bash "$installer" --check-dir-only --dir "${acl_parent}/private-state"
	done
elif [ "$(uname -s)" = Linux ]; then
	untrusted_parent="${test_root}/untrusted-owner"
	mkdir -m 0755 "$untrusted_parent"
	if [ "$(id -u)" -eq 0 ]; then
		chown 65534:65534 "$untrusted_parent"
		assert_fails_with 'owned by an untrusted user' bash "$installer" --check-dir-only --dir "${untrusted_parent}/private-state"
	elif command -v sudo >/dev/null 2>&1 && sudo -n chown 65534:65534 "$untrusted_parent"; then
		assert_fails_with 'owned by an untrusted user' bash "$installer" --check-dir-only --dir "${untrusted_parent}/private-state"
		sudo -n chown "$(id -u):$(id -g)" "$untrusted_parent"
	elif [ "${INSTALLER_REQUIRE_OWNERSHIP_TEST:-false}" = true ]; then
		printf 'ownership regression requires root or passwordless sudo\n' >&2
		exit 1
	fi
fi

fake_bin="${test_root}/fake-bin"
docker_log="${test_root}/docker.log"
volume_state="${test_root}/volume.state"
mkdir "$fake_bin"

cat >"${fake_bin}/docker" <<'FAKE_DOCKER'
#!/usr/bin/env bash
set -Eeuo pipefail
printf '%q ' "$@" >>"${FAKE_DOCKER_LOG:?}"
printf '\n' >>"${FAKE_DOCKER_LOG}"

if [ "${1:-}" = context ] && [ "${2:-}" = show ]; then
	printf 'default\n'
	exit 0
fi
if [ "${1:-}" = context ] && [ "${2:-}" = inspect ]; then
	printf '%s\n' "${FAKE_DOCKER_ENDPOINT:-unix:///var/run/docker.sock}"
	exit 0
fi
if [ "${1:-}" = info ]; then
	if [ "${2:-}" = --format ]; then
		printf '%s %s\n' "${FAKE_DOCKER_OS:-linux}" "${FAKE_DOCKER_ARCH:-x86_64}"
	fi
	exit 0
fi
if [ "${1:-}" = pull ]; then
	[ "${FAKE_DOCKER_PULL_FAIL:-false}" != true ]
	exit
fi
if [ "${1:-}" = image ] && [ "${2:-}" = inspect ]; then
	format="${4:-}"
	case "$format" in
	*org.opencontainers.image.title*) printf '%s\n' "${FAKE_IMAGE_TITLE:-AutoGPT Platform single-container}" ;;
	*org.opencontainers.image.source*) printf '%s\n' "${FAKE_IMAGE_SOURCE:-https://github.com/Significant-Gravitas/AutoGPT}" ;;
	*org.opencontainers.image.revision*) printf '%s\n' "${FAKE_IMAGE_REVISION:-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}" ;;
	'{{.Os}} {{.Architecture}}') printf '%s %s\n' "${FAKE_IMAGE_OS:-linux}" "${FAKE_IMAGE_ARCH:-amd64}" ;;
	*RepoDigests*)
		if [ "${FAKE_IMAGE_DIGESTS+x}" = x ]; then
			printf '%b' "$FAKE_IMAGE_DIGESTS"
		else
			printf '%b' 'significantgravitas/autogpt@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n'
		fi
		;;
	*) printf 'unexpected image inspect format: %s\n' "$format" >&2; exit 71 ;;
	esac
	exit 0
fi
if [ "${1:-}" = container ] && [ "${2:-}" = inspect ]; then
	[ "${FAKE_CONTAINER_EXISTS:-false}" = true ] || exit 1
	if [ "${3:-}" != --format ]; then exit 0; fi
	format="${4:-}"
	case "$format" in
	*com.agpt.installer.owner*)
		if command -v sha256sum >/dev/null 2>&1; then
			env_hash="$(sha256sum "${FAKE_INSTALL_DIR:?}/autogpt.env" | awk '{print $1}')"
		else
			env_hash="$(shasum -a 256 "${FAKE_INSTALL_DIR:?}/autogpt.env" | awk '{print $1}')"
		fi
		printf '%s\n' "${FAKE_CONTAINER_LABELS:-autogpt-platform-installer 1 $(cat "${FAKE_INSTALL_DIR}/install-id") ${env_hash}}"
		;;
	'{{.Config.Image}}')
		printf '%s\n' "${FAKE_CONTAINER_IMAGE:-docker.io/significantgravitas/autogpt@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb}"
		;;
	*HostConfig.Privileged*)
		printf '%s\n' "${FAKE_CONTAINER_RUNTIME:-false|bridge|unless-stopped|360|2147483648|json-file|50m|5|1|nofile:65536:65536|1|volume:autogpt-platform-data:/data:true|1|3000/tcp=127.0.0.1:3000}"
		;;
	*Config.Entrypoint*)
		printf '%s\n' "${FAKE_CONTAINER_PROCESS:-[\"/usr/bin/tini\",\"--\",\"/opt/autogpt/single-container/entrypoint.sh\"]|[\"/usr/bin/supervisord\",\"-n\",\"-c\",\"/opt/autogpt/single-container/supervisor/supervisord.conf\"]}"
		;;
	'{{.State.Status}}') printf '%s\n' "${FAKE_CONTAINER_STATE:-running}" ;;
	*State.Health*) printf '%s\n' "${FAKE_CONTAINER_HEALTH:-healthy}" ;;
	*) printf 'unexpected container inspect format: %s\n' "$format" >&2; exit 72 ;;
	esac
	exit 0
fi
if [ "${1:-}" = volume ] && [ "${2:-}" = inspect ]; then
	if [ ! -s "${FAKE_VOLUME_STATE:?}" ]; then exit 1; fi
	if [ "${3:-}" = --format ]; then
		if [ "$(cat "$FAKE_VOLUME_STATE")" = unowned ]; then
			printf 'local <no value> <no value> <no value>\n'
		else
			printf 'local autogpt-platform-installer 1 %s\n' "$(cat "${FAKE_INSTALL_DIR:?}/install-id")"
		fi
	fi
	exit 0
fi
if [ "${1:-}" = volume ] && [ "${2:-}" = create ]; then
	printf 'owned\n' >"${FAKE_VOLUME_STATE:?}"
	printf 'autogpt-platform-data\n'
	exit 0
fi
if [ "${1:-}" = run ]; then
	if [ "${FAKE_DOCKER_RUN_FAIL:-false}" = true ]; then exit 1; fi
	printf 'fake-container-id\n'
	exit 0
fi
if [ "${1:-}" = start ]; then
	printf '%s\n' "${2:-}"
	exit 0
fi
if [ "${1:-}" = inspect ]; then
	printf 'running healthy\n'
	exit 0
fi
if [ "${1:-}" = logs ]; then exit 0; fi
printf 'unexpected fake docker invocation: %s\n' "$*" >&2
exit 70
FAKE_DOCKER
chmod +x "${fake_bin}/docker"

run_installer() {
	local install_dir="$1"
	shift
	FAKE_DOCKER_LOG="$docker_log" \
		FAKE_VOLUME_STATE="$volume_state" \
		FAKE_INSTALL_DIR="$install_dir" \
		PATH="${fake_bin}:${PATH}" \
		bash "$installer" --skip-preflight --dir "$install_dir" "$@"
}

assert_fake_fails() {
	local expected="$1" install_dir="$2"
	shift 2
	assert_fails_with "$expected" env \
		FAKE_DOCKER_LOG="$docker_log" \
		FAKE_VOLUME_STATE="$volume_state" \
		FAKE_INSTALL_DIR="$install_dir" \
		PATH="${fake_bin}:${PATH}" \
		"$@"
}

: >"$docker_log"
: >"$volume_state"
default_dir="${test_root}/default-install"
run_installer "$default_dir"
log="$(cat "$docker_log")"
assert_contains 'pull docker.io/significantgravitas/autogpt:latest' "$log"
assert_contains 'run --detach' "$log"
assert_contains 'docker.io/significantgravitas/autogpt@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' "$log"
assert_not_contains 'docker.io/significantgravitas/autogpt:latest' "$(grep '^run ' "$docker_log")"
assert_contains '--publish 127.0.0.1:3000:3000' "$log"
assert_contains '--restart unless-stopped' "$log"
assert_contains '--stop-timeout 360' "$log"
assert_contains '--volume autogpt-platform-data:/data' "$log"
assert_contains '--label com.agpt.installer.owner=autogpt-platform-installer' "$log"
assert_contains '--label com.agpt.installer.contract=1' "$log"
grep -Eq -- '--label com.agpt.installer.env-sha256=[0-9a-f]{64}' "$docker_log"
test "$(file_mode "$default_dir")" = 700
test "$(file_mode "${default_dir}/autogpt.env")" = 600
test "$(file_mode "${default_dir}/install-id")" = 600
test "$(file_mode "${default_dir}/install-state")" = 600
grep -Fxq 'AUTOGPT_PUBLIC_URL=http://localhost:3000' "${default_dir}/autogpt.env"

: >"$docker_log"
: >"$volume_state"
explicit_dir="${test_root}/explicit-install"
run_installer "$explicit_dir" --release v1.2.3
assert_contains 'pull docker.io/significantgravitas/autogpt:v1.2.3' "$(cat "$docker_log")"

remote_dir="${test_root}/remote-install"
assert_fake_fails 'Refusing non-local Docker endpoint: tcp://' "$remote_dir" \
	env DOCKER_HOST=tcp://example.invalid:2375 bash "$installer" --skip-preflight --dir "$remote_dir"

ssh_dir="${test_root}/ssh-install"
assert_fake_fails 'Refusing non-local Docker endpoint: ssh://' "$ssh_dir" \
	env FAKE_DOCKER_ENDPOINT=ssh://example.invalid bash "$installer" --skip-preflight --dir "$ssh_dir"

npipe_dir="${test_root}/npipe-install"
assert_fake_fails 'Refusing non-local Docker endpoint: npipe://' "$npipe_dir" \
	env FAKE_DOCKER_ENDPOINT=npipe:////./pipe/docker_engine bash "$installer" --skip-preflight --dir "$npipe_dir"

windows_dir="${test_root}/windows-daemon"
assert_fake_fails 'switch Docker Desktop to Linux containers' "$windows_dir" \
	env FAKE_DOCKER_OS=windows bash "$installer" --skip-preflight --dir "$windows_dir"

unsupported_arch_dir="${test_root}/unsupported-arch"
assert_fake_fails 'Unsupported Docker daemon architecture' "$unsupported_arch_dir" \
	env FAKE_DOCKER_ARCH=riscv64 bash "$installer" --skip-preflight --dir "$unsupported_arch_dir"

wrong_title_dir="${test_root}/wrong-title"
assert_fake_fails 'unexpected OCI title' "$wrong_title_dir" \
	env FAKE_IMAGE_TITLE='Not AutoGPT' bash "$installer" --skip-preflight --dir "$wrong_title_dir"

missing_digest_dir="${test_root}/missing-digest"
assert_fake_fails 'exactly one native repository digest; found 0' "$missing_digest_dir" \
	env FAKE_IMAGE_DIGESTS='' bash "$installer" --skip-preflight --dir "$missing_digest_dir"

multiple_digest_dir="${test_root}/multiple-digest"
multiple_digests='significantgravitas/autogpt@sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\nsignificantgravitas/autogpt@sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\n'
assert_fake_fails 'exactly one native repository digest; found 2' "$multiple_digest_dir" \
	env FAKE_IMAGE_DIGESTS="$multiple_digests" bash "$installer" --skip-preflight --dir "$multiple_digest_dir"

wrong_arch_dir="${test_root}/wrong-image-arch"
assert_fake_fails 'does not match Docker daemon architecture' "$wrong_arch_dir" \
	env FAKE_IMAGE_ARCH=arm64 bash "$installer" --skip-preflight --dir "$wrong_arch_dir"

collision_dir="${test_root}/container-collision"
assert_fake_fails 'installer ownership or configuration identity does not match' "$collision_dir" \
	env FAKE_CONTAINER_EXISTS=true FAKE_CONTAINER_LABELS='unowned 0 wrong wrong' bash "$installer" --skip-preflight --dir "$collision_dir"

: >"$docker_log"
printf 'owned\n' >"$volume_state"
FAKE_CONTAINER_EXISTS=true run_installer "$default_dir"
assert_not_contains 'run --detach' "$(cat "$docker_log")"
assert_contains 'container inspect --format' "$(cat "$docker_log")"

: >"$docker_log"
FAKE_CONTAINER_EXISTS=true FAKE_CONTAINER_STATE=exited run_installer "$default_dir"
assert_contains 'start autogpt' "$(cat "$docker_log")"

drift_dir="${test_root}/runtime-drift"
mkdir -m 0700 "$drift_dir"
printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n' >"${drift_dir}/install-id"
printf 'AUTOGPT_PUBLIC_URL=http://localhost:3000\n' >"${drift_dir}/autogpt.env"
assert_fake_fails 'critical runtime contract has drifted' "$drift_dir" \
	env FAKE_CONTAINER_EXISTS=true FAKE_CONTAINER_RUNTIME='true|host|no|0|0|none||||0||0||0|' bash "$installer" --skip-preflight --dir "$drift_dir"

missing_volume_dir="${test_root}/missing-prior-volume"
mkdir -m 0700 "$missing_volume_dir"
printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n' >"${missing_volume_dir}/install-id"
printf 'AUTOGPT_PUBLIC_URL=http://localhost:3000\n' >"${missing_volume_dir}/autogpt.env"
printf 'contract=1 lifecycle=established volume=autogpt-platform-data install-id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n' >"${missing_volume_dir}/install-state"
: >"$volume_state"
assert_fake_fails 'Installer state already exists but persistent volume autogpt-platform-data is missing' "$missing_volume_dir" \
	bash "$installer" --skip-preflight --dir "$missing_volume_dir"

retry_dir="${test_root}/retry-after-pull"
: >"$volume_state"
assert_fake_fails 'Published appliance image' "$retry_dir" \
	env FAKE_DOCKER_PULL_FAIL=true bash "$installer" --skip-preflight --dir "$retry_dir"
[ ! -e "${retry_dir}/install-state" ]
run_installer "$retry_dir"
test -f "${retry_dir}/install-state"
: >"$volume_state"
assert_fake_fails 'Installer state already exists but persistent volume autogpt-platform-data is missing' "$retry_dir" \
	bash "$installer" --skip-preflight --dir "$retry_dir"

run_failure_dir="${test_root}/run-failure"
: >"$volume_state"
assert_fake_fails 'Could not start the appliance' "$run_failure_dir" \
	env FAKE_DOCKER_RUN_FAIL=true bash "$installer" --skip-preflight --dir "$run_failure_dir"
test -f "${run_failure_dir}/install-state"
: >"$volume_state"
assert_fake_fails 'Installer state already exists but persistent volume autogpt-platform-data is missing' "$run_failure_dir" \
	bash "$installer" --skip-preflight --dir "$run_failure_dir"

unowned_volume_dir="${test_root}/unowned-volume"
printf 'unowned\n' >"$volume_state"
assert_fake_fails 'Refusing existing unowned or incompatible volume' "$unowned_volume_dir" \
	bash "$installer" --skip-preflight --dir "$unowned_volume_dir"

: >"$volume_state"
symlink_env_dir="${test_root}/symlink-env"
mkdir -m 0700 "$symlink_env_dir"
ln -s "${test_root}/outside.env" "${symlink_env_dir}/autogpt.env"
assert_fake_fails 'Refusing symlink for Runtime configuration' "$symlink_env_dir" \
	bash "$installer" --skip-preflight --dir "$symlink_env_dir"

hardlink_env_dir="${test_root}/hardlink-env"
mkdir -m 0700 "$hardlink_env_dir"
printf 'AUTOGPT_PUBLIC_URL=http://localhost:3000\n' >"${hardlink_env_dir}/autogpt.env"
ln "${hardlink_env_dir}/autogpt.env" "${test_root}/env-copy"
assert_fake_fails 'Runtime configuration has multiple hard links' "$hardlink_env_dir" \
	bash "$installer" --skip-preflight --dir "$hardlink_env_dir"

hardlink_state_dir="${test_root}/hardlink-state"
mkdir -m 0700 "$hardlink_state_dir"
printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n' >"${hardlink_state_dir}/install-id"
ln "${hardlink_state_dir}/install-id" "${test_root}/state-copy"
assert_fake_fails 'Installer identity has multiple hard links' "$hardlink_state_dir" \
	bash "$installer" --skip-preflight --dir "$hardlink_state_dir"

if [ "$(uname -s)" = Darwin ]; then
	acl_env_dir="${test_root}/acl-env"
	mkdir -m 0700 "$acl_env_dir"
	printf 'AUTOGPT_PUBLIC_URL=http://localhost:3000\n' >"${acl_env_dir}/autogpt.env"
	chmod +a 'everyone allow read,write' "${acl_env_dir}/autogpt.env"
	: >"$volume_state"
	run_installer "$acl_env_dir"
	# macOS exposes ACL entries through ls -le; find has no equivalent output.
	# shellcheck disable=SC2012
	if ls -lde "${acl_env_dir}/autogpt.env" | awk 'NR == 1 { exit substr($1, length($1), 1) == "+" ? 0 : 1 }'; then
		printf 'runtime configuration retained an extended ACL\n' >&2
		exit 1
	fi
fi

if [ "${INSTALLER_REAL_DOCKER_TEST:-false}" = true ]; then
	(
		set -Eeuo pipefail
		suffix="${BASHPID}"
		real_image="autogpt-installer-contract-smoke:${suffix}"
		real_volume="autogpt-installer-contract-smoke-${suffix}"
		real_container="autogpt-installer-contract-smoke-${suffix}"
		# ShellCheck cannot infer that EXIT invokes this callback.
		# shellcheck disable=SC2317,SC2329
		cleanup_real_contract() {
			docker container rm -f "$real_container" >/dev/null 2>&1 || true
			docker volume rm "$real_volume" >/dev/null 2>&1 || true
			docker image rm "$real_image" >/dev/null 2>&1 || true
		}
		trap cleanup_real_contract EXIT
		docker info >/dev/null
		tar -cf - --files-from /dev/null | docker import - "$real_image" >/dev/null
		docker volume create "$real_volume" >/dev/null
		docker create \
			--name "$real_container" \
			--restart unless-stopped \
			--stop-timeout 360 \
			--shm-size 2g \
			--ulimit nofile=65536:65536 \
			--log-driver json-file \
			--log-opt max-size=50m \
			--log-opt max-file=5 \
			--publish 127.0.0.1:3000:3000 \
			--volume "${real_volume}:/data" \
			"$real_image" /bin/true >/dev/null
		actual_contract="$(docker inspect --format '{{.HostConfig.Privileged}}|{{.HostConfig.NetworkMode}}|{{.HostConfig.RestartPolicy.Name}}|{{.Config.StopTimeout}}|{{.HostConfig.ShmSize}}|{{.HostConfig.LogConfig.Type}}|{{index .HostConfig.LogConfig.Config "max-size"}}|{{index .HostConfig.LogConfig.Config "max-file"}}|{{len .HostConfig.Ulimits}}|{{range .HostConfig.Ulimits}}{{.Name}}:{{.Soft}}:{{.Hard}}{{end}}|{{len .Mounts}}|{{range .Mounts}}{{.Type}}:{{.Name}}:{{.Destination}}:{{.RW}}{{end}}|{{len .HostConfig.PortBindings}}|{{range $key, $bindings := .HostConfig.PortBindings}}{{$key}}={{range $bindings}}{{.HostIp}}:{{.HostPort}}{{end}}{{end}}' "$real_container")"
		expected_contract="false|bridge|unless-stopped|360|2147483648|json-file|50m|5|1|nofile:65536:65536|1|volume:${real_volume}:/data:true|1|3000/tcp=127.0.0.1:3000"
		[ "$actual_contract" = "$expected_contract" ] || {
			printf 'real Docker runtime contract mismatch\nexpected: %s\nactual: %s\n' "$expected_contract" "$actual_contract" >&2
			exit 1
		}
	)
fi

printf 'installer shell tests passed\n'
