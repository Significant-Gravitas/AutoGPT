#!/usr/bin/env bash
# AutoGPT Platform release installer for Linux and macOS.
#
# Docker must already be installed, running locally, and using Linux containers.
# Download to a unique temporary file and execute only after a successful fetch:
#   ( installer="$(mktemp)" && trap 'rm -f "$installer"' EXIT &&
#     curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL -o "$installer" \
#       https://setup.agpt.co/install.sh && bash "$installer" )
set -Eeuo pipefail

DEPLOY_IMAGE='docker.io/significantgravitas/autogpt'
CONTAINER_NAME='autogpt'
DATA_VOLUME='autogpt-platform-data'
OWNER_LABEL_KEY='com.agpt.installer.owner'
OWNER_LABEL_VALUE='autogpt-platform-installer'
CONTRACT_LABEL_KEY='com.agpt.installer.contract'
CONTRACT_LABEL_VALUE='1'
INSTALL_ID_LABEL_KEY='com.agpt.installer.install-id'
ENV_HASH_LABEL_KEY='com.agpt.installer.env-sha256'
EXPECTED_TITLE='AutoGPT Platform single-container'
EXPECTED_SOURCE='https://github.com/Significant-Gravitas/AutoGPT'
MIN_RAM_GB=8
MIN_DISK_GB=25

CHECK_DIR_ONLY=false
RELEASE=''
DIR="${XDG_CONFIG_HOME:-${HOME}/.config}"
DIR="${DIR%/}/autogpt"
SKIP_PREFLIGHT=false
PREFLIGHT_ONLY=false
RESOLVE_ONLY=false
VER_REF=''
IMAGE=''
IMAGE_DIGEST=''
DAEMON_ARCH=''
INSTALL_ID=''
ENV_FILE=''
ENV_HASH=''
VOLUME_ESTABLISHED=false

if [ -t 1 ]; then
	C_G=$'\033[0;32m'
	C_Y=$'\033[1;33m'
	C_R=$'\033[0;31m'
	C_C=$'\033[0;36m'
	C_0=$'\033[0m'
else
	C_G=''
	C_Y=''
	C_R=''
	C_C=''
	C_0=''
fi

say() { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
ok() { printf '  %s[ OK ]%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '  %s[WARN]%s %s\n' "$C_Y" "$C_0" "$*"; }
step() { printf '\n%s==> %s%s\n' "$C_C" "$*" "$C_0"; }
die() {
	printf '\n%sError: %s%s\n' "$C_R" "$*" "$C_0" >&2
	exit 1
}

print_help() {
	cat <<'EOF'
AutoGPT Platform Installer (Linux + macOS)

Installs the published single-container appliance. A running local Docker
daemon configured for Linux containers is required.

Safe download and execution:
  ( installer="$(mktemp)" && trap 'rm -f "$installer"' EXIT &&
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL -o "$installer" \
      https://setup.agpt.co/install.sh && bash "$installer" )

Flags:
  --release <tag>      published vX.Y.Z image (default: latest)
  --dir <path>         private state directory (default: $XDG_CONFIG_HOME/autogpt
                       or $HOME/.config/autogpt)
  --skip-preflight     skip RAM and disk checks; Docker checks remain mandatory
  --preflight-only     validate the machine and install nothing
  --resolve-only       print the selected image and install nothing
  --help
EOF
	exit 0
}

need_val() {
	[ -n "${2:-}" ] || die "$1 needs a value (try --help)"
	case "$2" in
	-*) die "$1 needs a value, not another option: $2" ;;
	esac
}

while [ $# -gt 0 ]; do
	case "$1" in
	--release=*)
		RELEASE="${1#*=}"
		[ -n "$RELEASE" ] || die '--release needs a non-empty value (try --help)'
		;;
	--release)
		need_val "$1" "${2:-}"
		RELEASE="$2"
		shift
		;;
	--dir=*) DIR="${1#*=}" ;;
	--dir)
		need_val "$1" "${2:-}"
		DIR="$2"
		shift
		;;
	--skip-preflight) SKIP_PREFLIGHT=true ;;
	--preflight-only) PREFLIGHT_ONLY=true ;;
	--resolve-only) RESOLVE_ONLY=true ;;
	--check-dir-only) CHECK_DIR_ONLY=true ;;
	-h | --help) print_help ;;
	*) die "Unknown flag: $1 (try --help)" ;;
	esac
	shift
done

OS="$(uname -s)"
case "$OS" in
Linux*) OS_FAMILY=linux ;;
Darwin*) OS_FAMILY=macos ;;
*) die "Unsupported OS: $OS. This release installer supports Linux and macOS; Windows users should follow the manual setup guide." ;;
esac
ARCH="$(uname -m)"

release_image() {
	local release="$1" version
	if [[ "$release" =~ ^autogpt-platform-beta-v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
		version="${BASH_REMATCH[1]}"
	elif [[ "$release" =~ ^v([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
		version="${BASH_REMATCH[1]}"
	else
		die "Unsupported release tag: ${release}. Expected autogpt-platform-beta-vX.Y.Z or vX.Y.Z."
	fi
	printf '%s:v%s\n' "$DEPLOY_IMAGE" "$version"
}

resolve_version() {
	if [ -n "$RELEASE" ]; then
		VER_REF="$RELEASE"
		IMAGE="$(release_image "$RELEASE")"
	else
		VER_REF='latest'
		IMAGE="${DEPLOY_IMAGE}:latest"
	fi
}

path_owner_uid() {
	if stat -c '%u' "$1" >/dev/null 2>&1; then stat -c '%u' "$1"; else stat -f '%u' "$1"; fi
}

path_link_count() {
	if stat -c '%h' "$1" >/dev/null 2>&1; then stat -c '%h' "$1"; else stat -f '%l' "$1"; fi
}

path_mode() {
	if stat -c '%a' "$1" >/dev/null 2>&1; then stat -c '%a' "$1"; else stat -f '%Lp' "$1"; fi
}

path_is_group_or_other_writable() {
	local mode group other
	mode="$(path_mode "$1")"
	group="${mode: -2:1}"
	other="${mode: -1}"
	case "${group}${other}" in
	*[2367]*) return 0 ;;
	*) return 1 ;;
	esac
}

path_is_trusted_sticky_parent() {
	local owner
	[ -k "$1" ] || return 1
	owner="$(path_owner_uid "$1")"
	[ "$owner" = 0 ] || [ "$owner" = "$(id -u)" ]
}

path_has_unsafe_acl() {
	[ "$OS_FAMILY" = macos ] || return 1
	# macOS exposes ACL entries through ls -le; find has no equivalent output.
	# shellcheck disable=SC2012
	ls -lde "$1" 2>/dev/null | awk '
		NR > 1 && / allow / && /(write|add_file|add_subdirectory|delete|delete_child|writeattr|writeextattr|writeowner|writesecurity|chown)/ { unsafe = 1 }
		END { exit unsafe ? 0 : 1 }
	'
}

path_has_extended_acl() {
	[ "$OS_FAMILY" = macos ] || return 1
	# shellcheck disable=SC2012
	ls -lde "$1" 2>/dev/null | awk 'NR == 1 { exit substr($1, length($1), 1) == "+" ? 0 : 1 }'
}

walk_existing_path() {
	local requested="$1" current='' component owner
	local -a components
	IFS='/' read -r -a components <<<"${requested#/}"
	for component in "${components[@]}"; do
		case "$component" in
		'') continue ;;
		.) die "Install directory contains an unsafe '.' component: ${DIR}" ;;
		..) die "Install directory contains an unsafe '..' component: ${DIR}" ;;
		esac
		current="${current}/${component}"
		if [ -e "$current" ] || [ -L "$current" ]; then
			[ ! -L "$current" ] || die "Install directory contains a symlink component: ${current}"
			[ -d "$current" ] || die "Install directory component is not a directory: ${current}"
			owner="$(path_owner_uid "$current")"
			[ "$owner" = 0 ] || [ "$owner" = "$(id -u)" ] ||
				die "Install directory ancestor is owned by an untrusted user: ${current}"
			if path_is_group_or_other_writable "$current" && ! path_is_trusted_sticky_parent "$current"; then
				die "Install directory ancestor is writable by another user or group: ${current}"
			fi
			! path_has_unsafe_acl "$current" || die "Install directory ancestor has an unsafe writable ACL: ${current}"
		fi
	done
}

validate_install_directory() {
	local requested home_canonical probe
	[ -n "$DIR" ] || die 'Install directory cannot be empty.'
	case "$DIR" in
	*'//'*) die "Install directory contains redundant path separators: ${DIR}" ;;
	esac
	case "$DIR" in
	/*) requested="${DIR%/}" ;;
	*) requested="$(pwd -P)/${DIR%/}" ;;
	esac
	[ -n "$requested" ] || requested='/'
	walk_existing_path "$requested"

	home_canonical="$(cd "$HOME" && pwd -P)" || die "Could not resolve the current user's home directory."
	case "$requested" in
	/ | "$home_canonical" | /Applications | /bin | /boot | /dev | /etc | /home | /Library | /lib | /lib64 | /opt | /private | /proc | /root | /run | /sbin | /srv | /System | /tmp | /Users | /usr | /var | /Volumes)
		die "Refusing broad or system install directory: ${requested}"
		;;
	esac

	if [ -e "$requested" ] || [ -L "$requested" ]; then
		[ ! -L "$requested" ] || die "Install directory is a symlink: ${requested}"
		[ -d "$requested" ] || die "Install directory is not a directory: ${requested}"
		[ "$(path_owner_uid "$requested")" = "$(id -u)" ] ||
			die "Install directory is not owned by the current user: ${requested}"
		! path_is_group_or_other_writable "$requested" ||
			die "Install directory is writable by another user or group: ${requested}"
		[ "$(path_mode "$requested")" = 700 ] ||
			die "Existing install directory must already be private mode 0700: ${requested}"
		! path_has_extended_acl "$requested" ||
			die "Install directory has an extended ACL; use a new private directory: ${requested}"
	else
		probe="$(dirname "$requested")"
		while [ ! -e "$probe" ] && [ ! -L "$probe" ]; do probe="$(dirname "$probe")"; done
		[ ! -L "$probe" ] || die "Install directory parent is a symlink: ${probe}"
		if path_is_group_or_other_writable "$probe" && ! path_is_trusted_sticky_parent "$probe"; then
			die "Install directory parent is writable by another user or group: ${probe}"
		fi
	fi
	DIR="$requested"
}

prepare_install_directory() {
	local current='' component created=false
	local -a components
	validate_install_directory
	[ -d "$DIR" ] && return
	IFS='/' read -r -a components <<<"${DIR#/}"
	for component in "${components[@]}"; do
		[ -n "$component" ] || continue
		current="${current}/${component}"
		if [ -e "$current" ] || [ -L "$current" ]; then
			if [ -L "$current" ] || [ ! -d "$current" ]; then
				die "Install directory changed during setup: ${current}"
			fi
			continue
		fi
		if ! (umask 077 && mkdir "$current"); then
			die "Install directory changed during setup; refusing to reuse it: ${current}"
		fi
		created=true
	done
	[ "$created" = true ] || die "Install directory was not created safely: ${DIR}"
	validate_install_directory
	chmod 0700 "$DIR"
}

validate_private_file() {
	local file="$1" description="$2"
	[ ! -L "$file" ] || die "Refusing symlink for ${description}: ${file}"
	[ -f "$file" ] || die "${description} is not a regular file: ${file}"
	[ "$(path_owner_uid "$file")" = "$(id -u)" ] || die "${description} is not owned by the current user: ${file}"
	[ "$(path_link_count "$file")" = 1 ] || die "${description} has multiple hard links: ${file}"
	if path_has_extended_acl "$file"; then
		chmod -N "$file" || die "Could not remove the extended ACL from ${description}: ${file}"
		! path_has_extended_acl "$file" || die "${description} retains an extended ACL: ${file}"
	fi
	chmod 0600 "$file"
}

prepare_state() {
	local id_file="${DIR}/install-id" state_file="${DIR}/install-state"
	if [ ! -e "$id_file" ] && [ ! -L "$id_file" ]; then
		(umask 077 && od -An -N16 -tx1 /dev/urandom | tr -d ' \n' >"$id_file") ||
			die 'Could not create a private installer identity.'
	fi
	validate_private_file "$id_file" 'Installer identity'
	INSTALL_ID="$(<"$id_file")"
	[[ "$INSTALL_ID" =~ ^[0-9a-f]{32}$ ]] || die "Installer identity has invalid contents: ${id_file}"
	if [ -e "$state_file" ] || [ -L "$state_file" ]; then
		validate_private_file "$state_file" 'Installer lifecycle state'
		[ "$(<"$state_file")" = "contract=1 lifecycle=established volume=${DATA_VOLUME} install-id=${INSTALL_ID}" ] ||
			die "Installer lifecycle state has invalid contents: ${state_file}"
		VOLUME_ESTABLISHED=true
	fi
}

mark_install_established() {
	local state_file="${DIR}/install-state" temp_file
	if [ "$VOLUME_ESTABLISHED" = true ]; then return; fi
	temp_file="$(mktemp "${DIR}/.install-state.XXXXXX")" || die 'Could not create installer lifecycle state.'
	chmod 0600 "$temp_file"
	printf 'contract=1 lifecycle=established volume=%s install-id=%s\n' "$DATA_VOLUME" "$INSTALL_ID" >"$temp_file" || {
		rm -f "$temp_file"
		die 'Could not write installer lifecycle state.'
	}
	if [ -e "$state_file" ] || [ -L "$state_file" ]; then
		rm -f "$temp_file"
		die "Installer lifecycle state appeared during setup: ${state_file}"
	fi
	mv "$temp_file" "$state_file" || {
		rm -f "$temp_file"
		die 'Could not commit installer lifecycle state.'
	}
	validate_private_file "$state_file" 'Installer lifecycle state'
	VOLUME_ESTABLISHED=true
}

file_sha256() {
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$1" | awk '{print $1}'
	elif command -v shasum >/dev/null 2>&1; then
		shasum -a 256 "$1" | awk '{print $1}'
	else
		die 'sha256sum or shasum is required to bind the runtime configuration.'
	fi
}

prepare_environment() {
	ENV_FILE="${DIR}/autogpt.env"
	if [ ! -e "$ENV_FILE" ] && [ ! -L "$ENV_FILE" ]; then
		(umask 077 && printf '%s\n' \
			'# Private runtime configuration for the AutoGPT appliance.' \
			'AUTOGPT_PUBLIC_URL=http://localhost:3000' >"$ENV_FILE")
	fi
	validate_private_file "$ENV_FILE" 'Runtime configuration'
	ENV_HASH="$(file_sha256 "$ENV_FILE")"
	[[ "$ENV_HASH" =~ ^[0-9a-f]{64}$ ]] || die 'Could not hash the runtime configuration.'
	ok "Private runtime configuration: ${ENV_FILE}"
}

preflight() {
	step 'Hardware pre-flight checks'
	local ram_gb=0 ram_value='' target_dir free_gb
	info "Host OS: ${OS_FAMILY} (${ARCH}); the selected Docker daemon is validated separately."
	if [ "$OS_FAMILY" = linux ]; then
		ram_value="$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || true)"
		[[ "$ram_value" =~ ^[0-9]+$ ]] && ram_gb=$((ram_value / 1024 / 1024))
	else
		ram_value="$(sysctl -n hw.memsize 2>/dev/null || true)"
		[[ "$ram_value" =~ ^[0-9]+$ ]] && ram_gb=$((ram_value / 1024 / 1024 / 1024))
	fi
	if [ "$ram_gb" -eq 0 ]; then warn 'Could not determine installed RAM.'
	elif [ "$ram_gb" -lt "$MIN_RAM_GB" ]; then warn "Only ${ram_gb} GB RAM; ${MIN_RAM_GB} GB or more is recommended."
	else ok "${ram_gb} GB RAM"; fi

	target_dir="$DIR"
	while [ ! -d "$target_dir" ] && [ "$target_dir" != / ]; do target_dir="$(dirname "$target_dir")"; done
	free_gb="$(df -Pk "$target_dir" 2>/dev/null | awk 'NR==2{print int($4/1024/1024)}')"
	if [ "${free_gb:-0}" -lt "$MIN_DISK_GB" ]; then
		warn "Only ${free_gb:-0} GB free on the state filesystem. Docker may use another storage root; verify at least ${MIN_DISK_GB} GB there."
	else ok "${free_gb} GB free on the state filesystem (Docker storage is validated by the image pull)"; fi
}

normalize_arch() {
	case "$1" in
	amd64 | x86_64) printf 'amd64\n' ;;
	arm64 | aarch64) printf 'arm64\n' ;;
	*) return 1 ;;
	esac
}

validate_docker() {
	step 'Validating the local Docker daemon'
	local endpoint context daemon_os daemon_arch raw_arch
	command -v docker >/dev/null 2>&1 ||
		die 'Docker is required. Install it from https://docs.docker.com/engine/install/ or https://docs.docker.com/desktop/, start it, then retry.'
	if [ -n "${DOCKER_HOST:-}" ]; then
		endpoint="$DOCKER_HOST"
	else
		context="$(docker context show 2>/dev/null)" || die 'Could not read the selected Docker context.'
		endpoint="$(docker context inspect "$context" --format '{{(index .Endpoints "docker").Host}}' 2>/dev/null)" ||
			die 'Could not inspect the selected Docker context.'
	fi
	case "$endpoint" in
	unix://*) ;;
	*) die "Refusing non-local Docker endpoint: ${endpoint}. Select a local unix:// context." ;;
	esac
	docker info >/dev/null 2>&1 || die 'The local Docker daemon is not running or is not accessible. Start Docker and retry.'
	read -r daemon_os raw_arch < <(docker info --format '{{.OSType}} {{.Architecture}}') || die 'Could not inspect the Docker daemon platform.'
	[ "$daemon_os" = linux ] || die "Docker is using ${daemon_os:-an unknown mode}; switch Docker Desktop to Linux containers and retry."
	daemon_arch="$(normalize_arch "$raw_arch")" || die "Unsupported Docker daemon architecture: ${raw_arch:-unknown}. Use amd64 or arm64."
	DAEMON_ARCH="$daemon_arch"
	ok "Local Docker Linux daemon (${DAEMON_ARCH})"
}

validate_image() {
	local title source revision image_os raw_arch image_arch digest_output digest line count=0
	title="$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.title"}}' "$IMAGE")" || die 'Could not inspect appliance image title.'
	source="$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.source"}}' "$IMAGE")" || die 'Could not inspect appliance image source.'
	revision="$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$IMAGE")" || die 'Could not inspect appliance image revision.'
	[ "$title" = "$EXPECTED_TITLE" ] || die "Image ${IMAGE} is not the AutoGPT appliance (unexpected OCI title)."
	[ "$source" = "$EXPECTED_SOURCE" ] || die "Image ${IMAGE} is not from the expected AutoGPT source repository."
	[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || die "Image ${IMAGE} has an invalid source revision label."
	read -r image_os raw_arch < <(docker image inspect --format '{{.Os}} {{.Architecture}}' "$IMAGE") || die 'Could not inspect appliance image platform.'
	[ "$image_os" = linux ] || die "Image ${IMAGE} is not a Linux appliance image."
	image_arch="$(normalize_arch "$raw_arch")" || die "Image ${IMAGE} has unsupported architecture: ${raw_arch:-unknown}."
	[ "$image_arch" = "$DAEMON_ARCH" ] || die "Image architecture ${image_arch} does not match Docker daemon architecture ${DAEMON_ARCH}."

	digest_output="$(docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$IMAGE")" || die 'Could not resolve the appliance repository digest.'
	while IFS= read -r line; do
		[ -n "$line" ] || continue
		count=$((count + 1))
		digest="${line##*@}"
		case "$line" in
		docker.io/significantgravitas/autogpt@* | significantgravitas/autogpt@*) ;;
		*) die "Image ${IMAGE} resolved to an unexpected repository digest: ${line}" ;;
		esac
		[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]] || die "Image ${IMAGE} has an invalid repository digest."
	done <<<"$digest_output"
	[ "$count" -eq 1 ] || die "Image ${IMAGE} must resolve to exactly one native repository digest; found ${count}."
	IMAGE_DIGEST="${DEPLOY_IMAGE}@${digest}"
	ok "Appliance identity: ${IMAGE_DIGEST}"
}

prepare_image() {
	step "Pulling published appliance image: ${IMAGE}"
	docker pull "$IMAGE" || die "Published appliance image ${IMAGE} is unavailable. Retry after publication completes or choose an existing --release vX.Y.Z."
	validate_image
}

validate_volume() {
	local metadata
	if docker volume inspect "$DATA_VOLUME" >/dev/null 2>&1; then
		metadata="$(docker volume inspect --format "{{.Driver}} {{index .Labels \"${OWNER_LABEL_KEY}\"}} {{index .Labels \"${CONTRACT_LABEL_KEY}\"}} {{index .Labels \"${INSTALL_ID_LABEL_KEY}\"}}" "$DATA_VOLUME")" || die 'Could not inspect the persistent volume.'
		[ "$metadata" = "local ${OWNER_LABEL_VALUE} ${CONTRACT_LABEL_VALUE} ${INSTALL_ID}" ] ||
			die "Refusing existing unowned or incompatible volume: ${DATA_VOLUME}"
		return
	fi
	[ "$VOLUME_ESTABLISHED" = false ] ||
		die "Installer state already exists but persistent volume ${DATA_VOLUME} is missing. Refusing to create an empty replacement; restore the volume or use a new --dir."
	docker volume create --driver local \
		--label "${OWNER_LABEL_KEY}=${OWNER_LABEL_VALUE}" \
		--label "${CONTRACT_LABEL_KEY}=${CONTRACT_LABEL_VALUE}" \
		--label "${INSTALL_ID_LABEL_KEY}=${INSTALL_ID}" \
		"$DATA_VOLUME" >/dev/null || die 'Could not create the persistent volume.'
	metadata="$(docker volume inspect --format "{{.Driver}} {{index .Labels \"${OWNER_LABEL_KEY}\"}} {{index .Labels \"${CONTRACT_LABEL_KEY}\"}} {{index .Labels \"${INSTALL_ID_LABEL_KEY}\"}}" "$DATA_VOLUME")" || die 'Could not verify the persistent volume.'
	[ "$metadata" = "local ${OWNER_LABEL_VALUE} ${CONTRACT_LABEL_VALUE} ${INSTALL_ID}" ] ||
		die "Persistent volume ownership verification failed: ${DATA_VOLUME}"
}

inspect_container() {
	local format="$1"
	docker container inspect --format "$format" "$CONTAINER_NAME"
}

reuse_existing_container() {
	local labels image runtime process state health
	labels="$(inspect_container "{{index .Config.Labels \"${OWNER_LABEL_KEY}\"}} {{index .Config.Labels \"${CONTRACT_LABEL_KEY}\"}} {{index .Config.Labels \"${INSTALL_ID_LABEL_KEY}\"}} {{index .Config.Labels \"${ENV_HASH_LABEL_KEY}\"}}")" ||
		die "Could not inspect existing container ${CONTAINER_NAME}."
	[ "$labels" = "${OWNER_LABEL_VALUE} ${CONTRACT_LABEL_VALUE} ${INSTALL_ID} ${ENV_HASH}" ] ||
		die "Refusing existing container ${CONTAINER_NAME}: installer ownership or configuration identity does not match."

	image="$(inspect_container '{{.Config.Image}}')" || die "Could not inspect existing container image."
	[ "$image" = "$IMAGE_DIGEST" ] ||
		die "Refusing existing container ${CONTAINER_NAME}: immutable appliance digest does not match ${IMAGE_DIGEST}."

	runtime="$(inspect_container '{{.HostConfig.Privileged}}|{{.HostConfig.NetworkMode}}|{{.HostConfig.RestartPolicy.Name}}|{{.Config.StopTimeout}}|{{.HostConfig.ShmSize}}|{{.HostConfig.LogConfig.Type}}|{{index .HostConfig.LogConfig.Config "max-size"}}|{{index .HostConfig.LogConfig.Config "max-file"}}|{{len .HostConfig.Ulimits}}|{{range .HostConfig.Ulimits}}{{.Name}}:{{.Soft}}:{{.Hard}}{{end}}|{{len .Mounts}}|{{range .Mounts}}{{.Type}}:{{.Name}}:{{.Destination}}:{{.RW}}{{end}}|{{len .HostConfig.PortBindings}}|{{range $key, $bindings := .HostConfig.PortBindings}}{{$key}}={{range $bindings}}{{.HostIp}}:{{.HostPort}}{{end}}{{end}}')" ||
		die "Could not inspect existing container runtime contract."
	[ "$runtime" = 'false|bridge|unless-stopped|360|2147483648|json-file|50m|5|1|nofile:65536:65536|1|volume:autogpt-platform-data:/data:true|1|3000/tcp=127.0.0.1:3000' ] ||
		die "Refusing existing container ${CONTAINER_NAME}: critical runtime contract has drifted."

	process="$(inspect_container '{{json .Config.Entrypoint}}|{{json .Config.Cmd}}')" ||
		die "Could not inspect existing container process contract."
	[ "$process" = '["/usr/bin/tini","--","/opt/autogpt/single-container/entrypoint.sh"]|["/usr/bin/supervisord","-n","-c","/opt/autogpt/single-container/supervisor/supervisord.conf"]' ] ||
		die "Refusing existing container ${CONTAINER_NAME}: entrypoint or command was overridden."

	state="$(inspect_container '{{.State.Status}}')" || die "Could not inspect existing container state."
	case "$state" in
	running)
		health="$(inspect_container '{{if .State.Health}}{{.State.Health.Status}}{{end}}')" || die "Could not inspect existing container health."
		[ "$health" = healthy ] || die "Existing installer-owned container is running but not healthy (health=${health:-none})."
		ok 'Existing installer-owned appliance already matches the requested immutable contract.'
		;;
	created | exited)
		step 'Starting the existing installer-owned appliance'
		docker start "$CONTAINER_NAME" >/dev/null || die 'Could not start the existing installer-owned appliance.'
		;;
	*) die "Existing installer-owned container is in unsupported state: ${state}" ;;
	esac
}

start_appliance() {
	validate_volume
	mark_install_established
	if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
		reuse_existing_container
		return
	fi
	step 'Starting the AutoGPT appliance'
	docker run --detach \
		--name "$CONTAINER_NAME" \
		--label "${OWNER_LABEL_KEY}=${OWNER_LABEL_VALUE}" \
		--label "${CONTRACT_LABEL_KEY}=${CONTRACT_LABEL_VALUE}" \
		--label "${INSTALL_ID_LABEL_KEY}=${INSTALL_ID}" \
		--label "${ENV_HASH_LABEL_KEY}=${ENV_HASH}" \
		--restart unless-stopped \
		--stop-timeout 360 \
		--shm-size 2g \
		--ulimit nofile=65536:65536 \
		--log-driver json-file \
		--log-opt max-size=50m \
		--log-opt max-file=5 \
		--env-file "$ENV_FILE" \
		--publish 127.0.0.1:3000:3000 \
		--volume "${DATA_VOLUME}:/data" \
		"$IMAGE_DIGEST" >/dev/null || die 'Could not start the appliance. The persistent volume was not removed.'
}

wait_for_healthy() {
	step 'Waiting for the appliance to become healthy'
	local attempt state health
	for ((attempt = 1; attempt <= 180; attempt++)); do
		read -r state health < <(docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' "$CONTAINER_NAME")
		if [ "${state}:${health}" = running:healthy ]; then ok 'AutoGPT is healthy'; return; fi
		case "${state}:${health}" in
		exited:* | dead:* | removing:* | *:unhealthy)
			docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
			die "The appliance failed while starting (state=${state}, health=${health:-none})."
			;;
		esac
		sleep 5
	done
	docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
	die 'The appliance did not become healthy within 15 minutes.'
}

say ''
say '============================================='
say '         AutoGPT Platform Installer'
say '============================================='

if [ "$CHECK_DIR_ONLY" = true ]; then
	validate_install_directory
	info "Validated install directory: ${DIR}"
	exit 0
fi

resolve_version
if [ "$RESOLVE_ONLY" = true ]; then
	info "Selected release -> ${VER_REF} -> ${IMAGE}"
	exit 0
fi

validate_install_directory
if [ "$SKIP_PREFLIGHT" = true ]; then warn 'Hardware pre-flight skipped; Docker locality and appliance identity checks remain mandatory.'; else preflight; fi
validate_docker
if [ "$PREFLIGHT_ONLY" = true ]; then say ''; say '(--preflight-only: nothing was installed.)'; exit 0; fi

prepare_install_directory
prepare_state
prepare_environment
prepare_image
info "Selected release -> ${VER_REF} -> ${IMAGE_DIGEST}"
start_appliance
wait_for_healthy

say ''
say '============================================='
say '  AutoGPT is ready at http://localhost:3000'
say '============================================='
say "Configuration: ${ENV_FILE}"
say "Persistent data: Docker volume ${DATA_VOLUME}"
say "Immutable appliance: ${IMAGE_DIGEST}"
say 'Registration starts open. Create the intended account, promote it, then close signup as documented.'
