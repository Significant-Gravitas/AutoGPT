#!/usr/bin/env bash
# ============================================================================
# AutoGPT Platform - Ultimate Installer (Linux + macOS)
# ----------------------------------------------------------------------------
# Zero-prerequisite bootstrap: checks whether your machine CAN run AutoGPT,
# installs git + Docker if missing, fetches the repo at the version you pick,
# then hands off to setup-autogpt.sh to bring the stack up.
#
# One-liner:
#   curl -fsSL https://setup.agpt.co/install.sh | bash
#   curl -fsSL https://setup.agpt.co/install.sh | bash -s -- --dev --with-ollama
#
# Flags (parity with install.ps1):
#   --dev                install the dev branch (default: latest release)
#   --branch <name>      install a specific branch
#   --release <tag>      install a specific release tag
#   --dir <path>         install location (default: $HOME/AutoGPT)
#   --with-ollama        also install Ollama + wire local-LLM AutoPilot (no cloud keys)
#   --ollama-model NAME  model to pull (implies --with-ollama)
#   --ollama-host URL    use an existing Ollama at this URL (implies --with-ollama)
#   --skip-preflight     skip the capability checks (not recommended)
#   --help
# ============================================================================
set -euo pipefail

REPO_URL='https://github.com/Significant-Gravitas/AutoGPT.git'
REPO_API='https://api.github.com/repos/Significant-Gravitas/AutoGPT'
MIN_RAM_GB=8
MIN_DISK_GB=25

DEV=false; BRANCH=''; RELEASE=''; DIR="$HOME/AutoGPT"
WITH_OLLAMA=false; OLLAMA_MODEL=''; OLLAMA_HOST=''
SKIP_PREFLIGHT=false; PREFLIGHT_ONLY=false

# ---- tiny UI helpers (parity vocabulary with install.ps1) ----
if [ -t 1 ]; then C_G=$'\033[0;32m'; C_Y=$'\033[1;33m'; C_R=$'\033[0;31m'; C_C=$'\033[0;36m'; C_0=$'\033[0m'; else C_G=; C_Y=; C_R=; C_C=; C_0=; fi
say()  { printf '%s\n' "$*"; }
info() { printf '  %s\n' "$*"; }
ok()   { printf '  %s[ OK ]%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '  %s[WARN]%s %s\n' "$C_Y" "$C_0" "$*"; }
efail(){ printf '  %s[FAIL]%s %s\n' "$C_R" "$C_0" "$*"; }
step() { printf '\n%s==> %s%s\n' "$C_C" "$*" "$C_0"; }
die()  { printf '\n%sError: %s%s\n' "$C_R" "$*" "$C_0" >&2; exit 1; }

# Usage text as an inline heredoc — reading "$0" breaks under `curl | bash`
# (where $0 is "bash", not this script).
print_help() {
  cat <<'EOF'
AutoGPT Platform - Ultimate Installer (Linux + macOS)

Zero-prerequisite bootstrap: checks whether your machine CAN run AutoGPT,
installs git + Docker if missing, fetches the repo at the version you pick,
then hands off to setup-autogpt.sh to bring the stack up.

One-liner:
  curl -fsSL https://setup.agpt.co/install.sh | bash
  curl -fsSL https://setup.agpt.co/install.sh | bash -s -- --dev --with-ollama

Flags (parity with install.ps1):
  --dev                install the dev branch (default: latest release)
  --branch <name>      install a specific branch
  --release <tag>      install a specific release tag
  --dir <path>         install location (default: $HOME/AutoGPT)
  --with-ollama        also install Ollama + wire local-LLM AutoPilot (no cloud keys)
  --ollama-model NAME  model to pull (implies --with-ollama)
  --ollama-host URL    use an existing Ollama at this URL (implies --with-ollama)
  --skip-preflight     skip the capability checks (not recommended)
  --help
EOF
  exit 0
}

# ---- args: accept both "--flag=value" and "--flag value" (parity with install.ps1,
# where -Branch/-Release/-Dir take the next token) ----
need_val() { [ -n "${2:-}" ] || die "$1 needs a value (try --help)"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --dev)              DEV=true ;;
    --branch=*)         BRANCH="${1#*=}" ;;
    --branch)           need_val "$1" "${2:-}"; BRANCH="$2"; shift ;;
    --release=*)        RELEASE="${1#*=}" ;;
    --release)          need_val "$1" "${2:-}"; RELEASE="$2"; shift ;;
    --dir=*)            DIR="${1#*=}" ;;
    --dir)              need_val "$1" "${2:-}"; DIR="$2"; shift ;;
    --with-ollama)      WITH_OLLAMA=true ;;
    --ollama-model=*)   OLLAMA_MODEL="${1#*=}"; WITH_OLLAMA=true ;;
    --ollama-model)     need_val "$1" "${2:-}"; OLLAMA_MODEL="$2"; WITH_OLLAMA=true; shift ;;
    --ollama-host=*)    OLLAMA_HOST="${1#*=}";  WITH_OLLAMA=true ;;
    --ollama-host)      need_val "$1" "${2:-}"; OLLAMA_HOST="$2";  WITH_OLLAMA=true; shift ;;
    --skip-preflight)   SKIP_PREFLIGHT=true ;;
    --preflight-only)   PREFLIGHT_ONLY=true ;;
    -h|--help)          print_help ;;
    *) die "Unknown flag: $1 (try --help)" ;;
  esac
  shift
done

OS="$(uname -s)"
case "$OS" in
  Linux*)  OS_FAMILY=linux ;;
  Darwin*) OS_FAMILY=macos ;;
  *) die "Unsupported OS: $OS. This installer supports Linux and macOS; Windows users run install.ps1." ;;
esac
ARCH="$(uname -m)"

say ""
say "============================================="
say "      AutoGPT Platform - Ultimate Installer"
say "============================================="

SUDO=''
need_sudo() { if [ "$(id -u)" -ne 0 ]; then SUDO='sudo'; fi; }

# ---- resolve which version to install ----
VER_KIND=''; VER_REF=''
resolve_version() {
  if [ -n "$BRANCH" ];  then VER_KIND=branch; VER_REF="$BRANCH"; return; fi
  if [ -n "$RELEASE" ]; then VER_KIND=tag;    VER_REF="$RELEASE"; return; fi
  if [ "$DEV" = true ]; then VER_KIND=branch; VER_REF=dev; return; fi
  # default: latest release tag
  local tag=''
  tag="$(curl -fsSL "$REPO_API/releases/latest" 2>/dev/null | grep -m1 '"tag_name"' | sed -E 's/.*"tag_name": *"([^"]+)".*/\1/' || true)"
  if [ -n "$tag" ]; then VER_KIND=tag; VER_REF="$tag"
  else die "Couldn't determine the latest AutoGPT release (GitHub API unreachable?). Re-run when you're back online, or pass --dev for the development branch (or --release <tag> for a specific version)."; fi
}

# ============================================================================
# PRE-FLIGHT: "will AutoGPT actually run on this machine?"
# ============================================================================
preflight() {
  step "Pre-flight checks (can this machine run AutoGPT?)"
  local hard_fail=false

  info "OS: $OS_FAMILY ($ARCH)"

  # RAM
  local ram_gb=0
  if [ "$OS_FAMILY" = linux ]; then
    ram_gb=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))
  else
    ram_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
  fi
  if [ "$ram_gb" -lt "$MIN_RAM_GB" ]; then warn "Only ${ram_gb} GB RAM; the stack wants >= ${MIN_RAM_GB} GB. It may be slow / OOM."
  else ok "${ram_gb} GB RAM (>= ${MIN_RAM_GB} GB)"; fi

  # Disk (free GB on the install target's filesystem). df -Pk is POSIX on
  # both Linux and macOS; convert KB -> GB.
  local target_dir free_gb
  # Measure the install path itself when it already exists (a custom --dir on
  # its own mount), otherwise walk up to its nearest existing ancestor so e.g.
  # --dir /mnt/data/new/AutoGPT measures /mnt/data, not $HOME.
  target_dir="$DIR"
  while [ -n "$target_dir" ] && [ ! -d "$target_dir" ] && [ "$target_dir" != "/" ]; do
    target_dir="$(dirname "$target_dir")"
  done
  [ -d "$target_dir" ] || target_dir="$HOME"
  free_gb=$(df -Pk "$target_dir" 2>/dev/null | awk 'NR==2{print int($4/1024/1024)}')
  if [ "${free_gb:-0}" -lt "$MIN_DISK_GB" ]; then
    efail "Only ${free_gb} GB free; AutoGPT images + stack need ~${MIN_DISK_GB} GB."
    info "Fix: free up space and re-run."
    hard_fail=true
  else ok "${free_gb} GB free (>= ${MIN_DISK_GB} GB)"; fi

  if [ "$OS_FAMILY" = linux ]; then
    # Docker on Linux is native (no hypervisor needed). Just sanity-check.
    ok "Linux: Docker runs natively (no CPU virtualization required)"
    if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
      efail "Not root and 'sudo' is unavailable - can't install Docker/git."
      info "Fix: run as root, or install sudo."
      hard_fail=true
    else ok "Have root or sudo for package installs"; fi
  else
    # macOS: Docker Desktop runs a lightweight VM via Apple's Hypervisor framework (present on all supported Macs).
    local prod; prod="$(sw_vers -productVersion 2>/dev/null || echo 0)"
    ok "macOS $prod ($ARCH) - Docker Desktop supported"
    if [ "$ARCH" = "arm64" ]; then info "Apple Silicon detected - will use the arm64 Docker build."; fi
  fi

  if [ "$hard_fail" = true ]; then
    die "This machine can't run AutoGPT yet - resolve the [FAIL] item(s) above and re-run. Nothing was installed."
  fi
  ok "Pre-flight passed - this machine can run AutoGPT."
}

# ============================================================================
# PREREQS
# ============================================================================
install_git() {
  if command -v git >/dev/null 2>&1; then ok "git already installed"; return; fi
  step "Installing git"
  need_sudo
  if [ "$OS_FAMILY" = macos ]; then
    if command -v brew >/dev/null 2>&1; then brew install git
    else info "Triggering Apple Command Line Tools (git) install..."; xcode-select --install 2>/dev/null || true
         die "Finish the Command Line Tools popup, then re-run this installer."; fi
  else
    if   command -v apt-get >/dev/null 2>&1; then $SUDO apt-get update -y && $SUDO apt-get install -y git
    elif command -v dnf     >/dev/null 2>&1; then $SUDO dnf install -y git
    elif command -v yum     >/dev/null 2>&1; then $SUDO yum install -y git
    elif command -v pacman  >/dev/null 2>&1; then $SUDO pacman -Sy --noconfirm git
    elif command -v zypper  >/dev/null 2>&1; then $SUDO zypper install -y git
    else die "No supported package manager found - install git manually and re-run."; fi
  fi
  command -v git >/dev/null 2>&1 && ok "git installed" || die "git install failed."
}

install_curl() {
  if command -v curl >/dev/null 2>&1; then return; fi
  step "Installing curl"
  need_sudo
  if [ "$OS_FAMILY" = macos ]; then
    die "curl not found (it normally ships with macOS). Install it (e.g. 'brew install curl') and re-run."
  fi
  if   command -v apt-get >/dev/null 2>&1; then $SUDO apt-get update -y && $SUDO apt-get install -y curl
  elif command -v dnf     >/dev/null 2>&1; then $SUDO dnf install -y curl
  elif command -v yum     >/dev/null 2>&1; then $SUDO yum install -y curl
  elif command -v pacman  >/dev/null 2>&1; then $SUDO pacman -Sy --noconfirm curl
  elif command -v zypper  >/dev/null 2>&1; then $SUDO zypper install -y curl
  else die "No supported package manager found - install curl manually and re-run."; fi
  command -v curl >/dev/null 2>&1 && ok "curl installed" || die "curl install failed."
}

docker_ready() { command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; }

install_docker() {
  if docker_ready; then ok "Docker is installed and running"; return; fi
  if [ "$OS_FAMILY" = linux ]; then
    need_sudo
    if command -v docker >/dev/null 2>&1; then
      # Already installed, just not responding — the daemon is stopped. Start it
      # rather than re-running the get.docker.com installer.
      step "Starting the Docker daemon"
      $SUDO systemctl enable --now docker 2>/dev/null || $SUDO service docker start 2>/dev/null || true
    else
      step "Installing Docker Engine (get.docker.com)"
      curl -fsSL https://get.docker.com | $SUDO sh
      $SUDO systemctl enable --now docker 2>/dev/null || $SUDO service docker start 2>/dev/null || true
      if [ "$(id -u)" -ne 0 ]; then $SUDO usermod -aG docker "$USER" 2>/dev/null || true
        warn "Added you to the 'docker' group - a fresh login is needed for non-sudo docker. setup-autogpt.sh will use sudo as needed this run."; fi
    fi
    # Poll for the daemon — a slow systemd/service start isn't ready on the
    # first probe. docker_ready uses non-sudo `docker info`; also accept the
    # sudo path since group membership needs a fresh login this run.
    local i=0
    while ! { docker info >/dev/null 2>&1 || $SUDO docker info >/dev/null 2>&1; }; do
      i=$((i+1))
      if [ "$i" -ge 12 ]; then die "Docker installed but the daemon isn't running after ~60s. Start it (e.g. 'sudo systemctl start docker') and re-run."; fi
      sleep 5
    done
    ok "Docker Engine ready"
  else
    step "Installing Docker Desktop (macOS)"
    if [ -d "/Applications/Docker.app" ]; then info "Docker Desktop already installed."
    elif command -v brew >/dev/null 2>&1; then info "Installing via Homebrew..."; brew install --cask docker
    else
      local dmg="$HOME/Downloads/Docker.dmg"
      local dmg_arch url
      dmg_arch="$([ "$ARCH" = arm64 ] && echo arm64 || echo amd64)"
      url="https://desktop.docker.com/mac/main/$dmg_arch/Docker.dmg"
      info "Downloading Docker Desktop..."; curl -fsSL "$url" -o "$dmg"
      info "Mounting + copying to /Applications (needs your password)..."
      hdiutil attach "$dmg" -nobrowse -quiet
      # Always detach the mounted volume, even if the copy fails (set -e would
      # otherwise exit before the detach line and leave /Volumes/Docker mounted).
      if ! { $SUDO cp -R "/Volumes/Docker/Docker.app" /Applications/ 2>/dev/null || cp -R "/Volumes/Docker/Docker.app" /Applications/; }; then
        hdiutil detach "/Volumes/Docker" -quiet || true
        die "Failed to copy Docker.app to /Applications. Install Docker Desktop manually from https://www.docker.com/products/docker-desktop and re-run."
      fi
      hdiutil detach "/Volumes/Docker" -quiet || true
    fi
    info "Starting Docker Desktop..."; open -a Docker || true
    wait_docker
  fi
}

wait_docker() {
  step "Waiting for the Docker engine (first start can take a few minutes)"
  local i
  for i in $(seq 1 60); do
    if docker_ready; then ok "Docker engine is up"; return; fi
    sleep 10
    if [ "$i" -eq 12 ]; then warn "Still waiting - if Docker Desktop is asking you to accept terms / sign in, do that; then this will continue."; fi
  done
  die "Docker engine didn't come up. Open Docker Desktop once, finish any first-run prompts, then re-run."
}

# ============================================================================
# FETCH + HAND OFF
# ============================================================================
get_repo() {
  step "Fetching AutoGPT ($VER_KIND: $VER_REF) into $DIR"
  if [ -d "$DIR/.git" ]; then
    info "Repo already present - updating..."
    # Fully-qualify the ref: a bare name fetches remote branches only, so
    # default/--release installs (which are tags) would otherwise fail to update.
    local fetch_ref
    if [ "$VER_KIND" = tag ]; then fetch_ref="refs/tags/$VER_REF"; else fetch_ref="refs/heads/$VER_REF"; fi
    git -C "$DIR" fetch --depth 1 origin "$fetch_ref" && git -C "$DIR" checkout FETCH_HEAD
  elif [ -d "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ]; then
    # Non-empty but not a git checkout — usually a half-finished clone from an
    # interrupted run. A plain `git clone` into it would fail every rerun, so
    # tell the user exactly how to recover instead of failing cryptically.
    die "$DIR exists but is not a git checkout (leftover from an interrupted run?). Remove it and re-run:  rm -rf \"$DIR\""
  else
    mkdir -p "$(dirname "$DIR")"
    git clone --depth 1 --branch "$VER_REF" "$REPO_URL" "$DIR"
  fi
  [ -f "$DIR/autogpt_platform/installer/setup-autogpt.sh" ] || die "Clone/checkout failed - $DIR has no installer."
  ok "Repo ready at $DIR"
}

invoke_setup() {
  step "Handing off to setup-autogpt.sh (builds + starts the stack)"
  local args=()
  [ "$WITH_OLLAMA" = true ] && args+=(--with-ollama)
  [ -n "$OLLAMA_MODEL" ] && args+=("--ollama-model=$OLLAMA_MODEL")
  [ -n "$OLLAMA_HOST" ]  && args+=("--ollama-host=$OLLAMA_HOST")
  info "Running: setup-autogpt.sh ${args[*]:-}"
  # Run from inside the checkout: setup-autogpt.sh's detect_repo() keys off $PWD
  # (not --dir), so launching it from elsewhere makes it try to git-clone a
  # fresh copy into $PWD/AutoGPT instead of using the repo we just fetched.
  cd "$DIR" || die "Cannot enter $DIR"
  # Guard the empty-array case (set -u + bash <4.4 errors on "${a[@]}" when empty)
  if [ "${#args[@]}" -gt 0 ]; then
    bash "autogpt_platform/installer/setup-autogpt.sh" "${args[@]}"
  else
    bash "autogpt_platform/installer/setup-autogpt.sh"
  fi
}

# ============================================================================
# MAIN
# ============================================================================
if [ "$SKIP_PREFLIGHT" = true ]; then warn "Pre-flight skipped (--skip-preflight)."; else preflight; fi
if [ "$PREFLIGHT_ONLY" = true ]; then say ""; say "(--preflight-only: stopping before any install.)"; exit 0; fi
install_curl   # resolve_version + get.docker.com both rely on curl
install_git
resolve_version
info "Selected version -> $VER_KIND: $VER_REF"
install_docker
get_repo
invoke_setup

say ""
say "============================================="
say "  Done. AutoGPT should be at http://localhost:3000"
say "============================================="
