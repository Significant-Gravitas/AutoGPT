#!/bin/bash

# ------------------------------------------------------------------------------
# AutoGPT Setup Script (Linux + macOS)
# ------------------------------------------------------------------------------
# This script automates the installation and setup of AutoGPT.
# It checks prerequisites, clones the repository, and starts all services.
# Windows users: see setup-autogpt.bat.
#
# Optional flags:
#   --with-ollama          Also install Ollama, pull a default chat model, and
#                          wire backend/.env so AutoPilot runs without any
#                          cloud API keys (CHAT_USE_LOCAL=true). See
#                          docs/platform/copilot-local-llm.md.
#   --ollama-model=NAME    Model to pull (default: hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M).
#   --ollama-host=URL      Use an existing Ollama at this URL instead of
#                          installing one locally. Skips the Ollama install
#                          but still writes the CHAT_USE_LOCAL .env entries.
#                          Example: --ollama-host=http://gpu-rig.lab:11434
# ------------------------------------------------------------------------------

# --- Global Variables ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Variables
REPO_DIR=""
CLONE_NEEDED=false
DOCKER_CMD="docker"
DOCKER_COMPOSE_CMD="docker compose"
LOG_FILE=""
WITH_OLLAMA=false
OLLAMA_MODEL="hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M"
OLLAMA_HOST_URL=""

# OS-family detection — Linux and macOS install Ollama very differently
# (systemd unit + curl-pipe-sh vs launchd-managed brew/dmg app), bind the
# host-to-container reachable IP differently (LAN IP / bridge vs
# host.docker.internal), and set Ollama env differently (drop-in
# /etc/systemd vs launchctl setenv). One central detector beats sprinkling
# ``uname -s`` checks through every helper.
OS_FAMILY="unknown"
case "$(uname -s)" in
    Linux*)  OS_FAMILY="linux" ;;
    Darwin*) OS_FAMILY="macos" ;;
    *)       OS_FAMILY="unknown" ;;
esac

for arg in "$@"; do
    case "$arg" in
        --with-ollama)        WITH_OLLAMA=true ;;
        --ollama-model=*)     OLLAMA_MODEL="${arg#*=}"; WITH_OLLAMA=true ;;
        --ollama-host=*)      OLLAMA_HOST_URL="${arg#*=}"; WITH_OLLAMA=true ;;
        -h|--help)            sed -n '4,19p' "$0"; exit 0 ;;
        *) echo "Unknown flag: $arg" >&2; exit 2 ;;
    esac
done

print_color() {
    printf "${!1}%s${NC}\n" "$2"
}

print_banner() {
    print_color "BLUE" "
       d8888          888             .d8888b.  8888888b. 88888888888 
      d88888          888            d88P  Y88b 888   Y88b    888     
     d88P888          888            888    888 888    888    888     
    d88P 888 888  888 888888 .d88b.  888        888   d88P    888     
   d88P  888 888  888 888   d88\"\"88b 888  88888 8888888P\"     888     
  d88P   888 888  888 888   888  888 888    888 888           888     
 d8888888888 Y88b 888 Y88b. Y88..88P Y88b  d88P 888           888     
d88P     888  \"Y88888  \"Y888 \"Y88P\"   \"Y8888P88 888           888     
"
}

handle_error() {
    print_color "RED" "Error: $1"
    if [ -n "$LOG_FILE" ] && [ -f "$LOG_FILE" ]; then
        print_color "RED" "Check log file for details: $LOG_FILE"
    fi
    exit 1
}

check_prerequisites() {
    print_color "BLUE" "Checking prerequisites..."

    if [ "$OS_FAMILY" = "unknown" ]; then
        handle_error "Unsupported OS: $(uname -s). This script supports Linux and macOS — Windows users should run setup-autogpt.bat."
    fi
    print_color "GREEN" "✓ OS family: $OS_FAMILY"

    if ! command -v git &> /dev/null; then
        handle_error "Git is not installed. Please install it and try again."
    else
        print_color "GREEN" "✓ Git is installed"
    fi

    if ! command -v docker &> /dev/null; then
        handle_error "Docker is not installed. Please install it and try again."
    else
        print_color "GREEN" "✓ Docker is installed"
    fi

    # ``--with-ollama`` shells out to curl for: the Ollama install-script
    # download, version/tags/pull probes against local + remote backends,
    # and the post-restart readiness loop. A minimal host without curl
    # would otherwise fail mid-setup with an opaque ``command not found``;
    # surfacing it here keeps the prerequisite contract in one place.
    if [ "$WITH_OLLAMA" = true ] && ! command -v curl &> /dev/null; then
        handle_error "curl is not installed but --with-ollama needs it (for the Ollama install + API probes). Install curl and re-run."
    fi
    
    if ! docker info &> /dev/null; then
        print_color "YELLOW" "Using sudo for Docker commands..."
        DOCKER_CMD="sudo docker"
        DOCKER_COMPOSE_CMD="sudo docker compose"
    fi
    
    print_color "GREEN" "All prerequisites installed!"
}

detect_repo() {
    if [[ "$PWD" == */autogpt_platform/installer ]]; then
        if [[ -d "../../.git" ]]; then
            REPO_DIR="$(cd ../..; pwd)"
            cd ../.. || handle_error "Failed to navigate to repo root"
            print_color "GREEN" "Using existing AutoGPT repository."
        else
            CLONE_NEEDED=true
            REPO_DIR="$(pwd)/AutoGPT"
        fi
    elif [[ -d ".git" && -d "autogpt_platform/installer" ]]; then
        REPO_DIR="$PWD"
        print_color "GREEN" "Using existing AutoGPT repository."
    else
        CLONE_NEEDED=true
        REPO_DIR="$(pwd)/AutoGPT"
    fi
}

clone_repo() {
    if [ "$CLONE_NEEDED" = true ]; then
        print_color "BLUE" "Cloning AutoGPT repository..."
        git clone https://github.com/Significant-Gravitas/AutoGPT.git "$REPO_DIR" || handle_error "Failed to clone repository"
        print_color "GREEN" "Repository cloned successfully."
    fi
}

bootstrap_ollama() {
    # Only runs when --with-ollama is set. Splits installing Ollama (which
    # we skip if --ollama-host points at an existing one) from pulling the
    # model (which still happens against the remote in case the operator
    # forgot). The systemd drop-in binds Ollama to all interfaces so
    # docker containers on the same host can reach it via the bridge —
    # without it, OLLAMA_HOST defaults to 127.0.0.1 and containers see
    # connection refused.
    if [ -n "$OLLAMA_HOST_URL" ]; then
        # Normalize: strip trailing slash and an optional ``/v1`` so that
        # ``--ollama-host`` accepts either the Ollama root
        # (``http://host:11434``) or a copy-pasted ``CHAT_BASE_URL`` value
        # (``http://host:11434/v1``). Without this, an operator passing the
        # latter would have us probe ``/v1/api/version`` (404) and write
        # ``CHAT_BASE_URL=…/v1/v1`` into the .env.
        local OLLAMA_ROOT="${OLLAMA_HOST_URL%/}"
        OLLAMA_ROOT="${OLLAMA_ROOT%/v1}"
        print_color "BLUE" "Using existing Ollama at $OLLAMA_ROOT"
        if ! curl -sf "${OLLAMA_ROOT}/api/version" > /dev/null; then
            handle_error "Cannot reach Ollama at $OLLAMA_ROOT — is it running and listening on 0.0.0.0?"
        fi
        # Validate the configured model is present on the remote host;
        # pull it if not. Without this, the install reports "ready" but
        # the first chat turn fails with "model not found" — a setup
        # script that returns 0 should mean the platform is *usable*.
        # ``grep -F`` (fixed string) so model names containing regex
        # metacharacters (``hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M`` has ``.`` and
        # ``:``) match literally, not as wildcards.
        if ! curl -sf "${OLLAMA_ROOT}/api/tags" \
            | grep -Fq "\"name\":\"$OLLAMA_MODEL\""; then
            print_color "BLUE" "Model '$OLLAMA_MODEL' missing on remote — requesting pull..."
            # ``/api/pull`` returns HTTP 200 immediately and streams NDJSON
            # progress lines. A genuine failure (registry 404, network blip
            # mid-download) appears as a JSON object with an ``error`` field
            # in the response body — NOT as a non-2xx status — so ``curl -f``
            # alone reports success and the install lies about being usable.
            # We capture the stream and grep for an explicit success or
            # error frame to surface the truth.
            local pull_log
            pull_log=$(curl -sf -N "${OLLAMA_ROOT}/api/pull" \
                -H "Content-Type: application/json" \
                -d "{\"name\":\"$OLLAMA_MODEL\"}") \
                || handle_error "Pull request to $OLLAMA_ROOT/api/pull failed"
            if grep -q '"error"' <<< "$pull_log"; then
                local err
                err=$(grep -m1 '"error"' <<< "$pull_log")
                handle_error "Pull of $OLLAMA_MODEL failed: $err"
            fi
            if ! grep -q '"status":"success"' <<< "$pull_log"; then
                handle_error "Pull of $OLLAMA_MODEL did not report success — last frame: $(tail -1 <<< "$pull_log")"
            fi
            print_color "GREEN" "✓ Pulled $OLLAMA_MODEL on remote"
        else
            print_color "GREEN" "✓ Model $OLLAMA_MODEL present on remote"
        fi
        # Stash the normalized root so write_local_env doesn't have to
        # repeat the trim. Plain global var is fine — bash functions
        # share the parent's scope unless ``local`` is used.
        OLLAMA_HOST_URL="$OLLAMA_ROOT"
        return
    fi
    if [ "$OS_FAMILY" = "macos" ]; then
        _bootstrap_ollama_macos
    else
        _bootstrap_ollama_linux
    fi
}

_bootstrap_ollama_linux() {
    if ! command -v ollama &> /dev/null; then
        print_color "BLUE" "Installing Ollama (https://ollama.com/install.sh)..."
        curl -fsSL https://ollama.com/install.sh | sh || handle_error "Ollama install failed"
    else
        print_color "GREEN" "✓ Ollama already installed ($(ollama --version 2>&1 | head -1))"
    fi
    # systemd drop-in: bind to all interfaces (so containers can reach it)
    # AND raise the context window from Ollama's 4 k default. Without
    # OLLAMA_CONTEXT_LENGTH, Ollama silently truncates AutoPilot's ~8 k
    # system prompt to 4 k regardless of the model's advertised window
    # — the OpenAI shim does NOT honor `options.num_ctx` in the request
    # body (ollama/ollama#2714), only the env / Modelfile knob does.
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    sudo tee /etc/systemd/system/ollama.service.d/host.conf > /dev/null <<'OLLAMA_DROPIN'
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_CONTEXT_LENGTH=32768"
OLLAMA_DROPIN
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    for _ in $(seq 1 20); do
        curl -sf http://localhost:11434/api/version >/dev/null && break
        sleep 1
    done
    print_color "BLUE" "Pulling model: $OLLAMA_MODEL (this may take several minutes)..."
    ollama pull "$OLLAMA_MODEL" || handle_error "Failed to pull $OLLAMA_MODEL"
    print_color "GREEN" "✓ Ollama ready: http://localhost:11434"
}

_bootstrap_ollama_macos() {
    # macOS Ollama is a launchd-managed .app, not a systemd unit. The
    # canonical install paths are (1) ``brew install ollama`` (formula
    # = headless server; cask = full .app), or (2) the official .dmg
    # from ollama.com. We prefer brew when present because it's
    # scriptable; otherwise we point the operator at the .dmg.
    # Install the official Ollama via the Homebrew *cask* (ollama-app), NOT the
    # ``ollama`` formula. The formula's darwin-arm64 bottle ships the server
    # without the llama-server runner (libexec/lib/ollama holds only the MLX
    # metal lib), so it pulls models fine but 500s on every GGUF inference with
    # "llama-server binary not found". The cask is the full Ollama.app and
    # bundles the runner at Contents/Resources/llama-server.
    local runner="/Applications/Ollama.app/Contents/Resources/llama-server"
    if [ -x "$runner" ]; then
        print_color "GREEN" "✓ Ollama already installed with runner ($(ollama --version 2>&1 | head -1))"
    elif command -v brew &> /dev/null; then
        # A runner-less formula install leaves an ``ollama`` on PATH that can't
        # run models — replace it with the cask rather than skipping on
        # ``command -v``.
        if brew list --formula ollama &> /dev/null; then
            print_color "YELLOW" "Replacing runner-less Ollama formula with the official cask..."
            brew uninstall ollama &> /dev/null || true
        fi
        print_color "BLUE" "Installing Ollama (official app) via Homebrew cask..."
        brew install --cask ollama-app || handle_error "brew install --cask ollama-app failed"
    else
        handle_error "Ollama is not installed and Homebrew is not available. Install Homebrew (https://brew.sh) and re-run, or download Ollama from https://ollama.com/download/mac and re-run with --ollama-host=http://localhost:11434."
    fi
    # Set Ollama env globally for launchd-spawned processes. On macOS,
    # ``launchctl setenv`` is the only knob that survives Ollama.app
    # being relaunched by launchd — exporting in shell won't reach it.
    # **But ``launchctl setenv`` does NOT update an already-running
    # process** — it writes to launchd's env table, which is read at
    # next spawn. If Ollama.app is already running (tray icon visible,
    # carrying its old OLLAMA_HOST=127.0.0.1 default), the script will
    # otherwise report success while the running app continues to
    # truncate context at 4 k and bind only to loopback. Kill the
    # existing instance(s) before re-launching so the new env takes
    # effect.
    launchctl setenv OLLAMA_HOST "0.0.0.0:11434" \
        || handle_error "launchctl setenv OLLAMA_HOST failed"
    launchctl setenv OLLAMA_CONTEXT_LENGTH "32768" \
        || handle_error "launchctl setenv OLLAMA_CONTEXT_LENGTH failed"
    # Stop any currently-running Ollama (tray app, brew service, raw
    # ``ollama serve``). ``pkill`` matches by name; ``|| true`` so the
    # script doesn't abort when nothing is running.
    osascript -e 'quit app "Ollama"' >/dev/null 2>&1 || true
    pkill -f "ollama serve" >/dev/null 2>&1 || true
    if brew services list 2>/dev/null | grep -q '^ollama'; then
        brew services stop ollama >/dev/null 2>&1 || true
    fi
    # Wait briefly for the port to release.
    for _ in $(seq 1 5); do
        if ! lsof -nP -iTCP:11434 -sTCP:LISTEN >/dev/null 2>&1; then break; fi
        sleep 1
    done
    # Restart Ollama headlessly. ``launchctl setenv`` above only reaches
    # launchd-spawned GUI launches of Ollama.app — a shell-spawned
    # ``ollama serve`` inherits THIS shell's env, not launchd's, so it would
    # otherwise bind 127.0.0.1 and truncate context at Ollama's 4k default.
    # Export the same vars on the serve invocation so the headless server
    # honors them too. ``disown`` so the background job survives this script's
    # exit even in shells with ``shopt -s huponexit`` (login shells, some CI).
    OLLAMA_HOST="0.0.0.0:11434" OLLAMA_CONTEXT_LENGTH="32768" \
        nohup ollama serve >/dev/null 2>&1 &
    disown 2>/dev/null || true
    for _ in $(seq 1 20); do
        curl -sf http://localhost:11434/api/version >/dev/null && break
        sleep 1
    done
    if ! curl -sf http://localhost:11434/api/version >/dev/null; then
        handle_error "Ollama did not become reachable on localhost:11434. If you installed via the .dmg, open Ollama.app once to grant it network permissions, then re-run."
    fi
    print_color "BLUE" "Pulling model: $OLLAMA_MODEL (this may take several minutes)..."
    ollama pull "$OLLAMA_MODEL" || handle_error "Failed to pull $OLLAMA_MODEL"
    print_color "GREEN" "✓ Ollama ready: http://localhost:11434"
}

write_local_env() {
    # Wire backend/.env so the new ChatConfig.local transport activates and
    # AutoPilot routes through Ollama with no cloud API keys. Uses the host
    # LAN IP (or the explicit --ollama-host URL) so containers on Linux
    # can reach Ollama without docker-compose extra_hosts gymnastics.
    cd "$REPO_DIR/autogpt_platform/backend" || handle_error "no backend dir"
    [ -f .env ] || cp .env.default .env
    local host_url
    if [ -n "$OLLAMA_HOST_URL" ]; then
        # ``bootstrap_ollama`` already stripped the trailing slash + any
        # ``/v1`` from the operator-supplied URL; trust the normalized form.
        host_url="$OLLAMA_HOST_URL"
    elif [ "$OS_FAMILY" = "macos" ]; then
        # On macOS Docker Desktop auto-injects ``host.docker.internal``
        # into every container's /etc/hosts. Colima, Rancher Desktop,
        # and Lima/nerdctl do NOT — those runtimes need the host's LAN
        # IP instead, otherwise every container DNS lookup fails with
        # ``getaddrinfo ENOTFOUND host.docker.internal`` and the
        # installer would report success while the platform is unusable.
        # Detect by probing ``docker context show`` for the Desktop
        # context name; fall back to a LAN IP for non-Desktop runtimes.
        local docker_ctx
        docker_ctx=$(docker context show 2>/dev/null || echo "")
        if [ "$docker_ctx" = "desktop-linux" ] || [ "$docker_ctx" = "default" ]; then
            host_url="http://host.docker.internal:11434"
        else
            # Colima / Rancher Desktop / Lima — get the LAN IP that the
            # VM can route to. ``ipconfig getifaddr en0`` is the canonical
            # macOS one-liner for "the active interface's IPv4".
            local host_ip
            host_ip=$(ipconfig getifaddr en0 2>/dev/null \
                      || ipconfig getifaddr en1 2>/dev/null \
                      || echo "")
            if [ -z "$host_ip" ]; then
                handle_error "Could not detect a LAN IP for non-Docker-Desktop runtime ($docker_ctx). Re-run with --ollama-host=http://<your-LAN-IP>:11434."
            fi
            host_url="http://${host_ip}:11434"
        fi
    else
        # Linux: containers can't reach the host via host.docker.internal
        # unless compose wires it. ``hostname -I`` (Linux-only) gives the
        # first non-loopback IPv4, which the bridge network always reaches.
        # We fall back to 127.0.0.1 only when -I returns nothing
        # (e.g. a host with no LAN IPs); in that case the operator will
        # need to override CHAT_BASE_URL by hand.
        local host_ip
        host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
        : "${host_ip:=127.0.0.1}"
        host_url="http://${host_ip}:11434"
    fi
    # Strip ONLY the bounded block we wrote previously (idempotent re-run).
    # Earlier draft used '/MARKER/,$d' which deletes from marker to EOF —
    # any user-added lines after the block were silently lost on every
    # re-run. The start/end markers below let sed range-delete the exact
    # block we own and nothing else.
    local START_MARKER='# === Local-LLM AutoPilot wiring (added by setup-autogpt.sh --with-ollama) ==='
    local END_MARKER='# === End Local-LLM AutoPilot wiring ==='
    if grep -qF "$START_MARKER" .env; then
        # Use ``/`` as the address-pattern delimiter — the markers contain
        # ``#`` so picking ``#`` would terminate the regex on the very first
        # marker character, leaving sed to match every line and delete the
        # whole file. ``/`` is safe because the markers contain no ``/``.
        # Escape BRE metacharacters in the markers (``.`` from
        # ``setup-autogpt.sh``, etc.) so the address pattern matches the
        # literal marker line and nothing that happens to differ by one
        # character. ``grep -qF`` above ensures we only run when the exact
        # marker is present.
        local START_RE END_RE
        START_RE=$(printf '%s\n' "$START_MARKER" | sed 's/[].[\*^$/]/\\&/g')
        END_RE=$(printf '%s\n' "$END_MARKER" | sed 's/[].[\*^$/]/\\&/g')
        # ``sed -i`` is incompatible between GNU (Linux) and BSD (macOS):
        # GNU treats ``-i`` alone as in-place; BSD requires an extension
        # argument (``-i ''``) or it eats the next arg as the suffix.
        # ``sed -i.bak`` works on both — we delete the .bak afterwards.
        sed -i.bak "/$START_RE/,/$END_RE/d" .env && rm -f .env.bak
    fi
    {
        echo
        echo "$START_MARKER"
        echo "# See docs/platform/copilot-local-llm.md for the full reference."
        echo "CHAT_USE_LOCAL=true"
        echo "CHAT_BASE_URL=${host_url}/v1"
        echo "CHAT_API_KEY=ollama"
        echo "CHAT_FAST_STANDARD_MODEL=$OLLAMA_MODEL"
        echo "CHAT_FAST_ADVANCED_MODEL=$OLLAMA_MODEL"
        # title_model + simulation_model auto-derive from fast_standard_model
        # under the local transport — no need to set them explicitly.
        # OLLAMA_HOST is the block-layer LLM (separate from the chat path);
        # set it too so the AI Text Generator block points at the same backend.
        echo "OLLAMA_HOST=${host_url}"
        echo "$END_MARKER"
    } >> .env
    cd ..
    print_color "GREEN" "✓ wrote backend/.env (CHAT_USE_LOCAL=true, Ollama at $host_url)"
}

run_docker() {
    cd "$REPO_DIR/autogpt_platform" || handle_error "Failed to navigate to autogpt_platform"
    
    print_color "BLUE" "Starting AutoGPT services with Docker Compose..."
    print_color "YELLOW" "This may take a few minutes on first run..."
    echo
    
    mkdir -p logs
    LOG_FILE="$REPO_DIR/autogpt_platform/logs/docker_setup.log"
    
    if $DOCKER_COMPOSE_CMD up -d > "$LOG_FILE" 2>&1; then
        print_color "GREEN" "✓ Services started successfully!"
    else
        print_color "RED" "Docker compose failed. Check log file for details: $LOG_FILE"
        print_color "YELLOW" "Common issues:"
        print_color "YELLOW" "- Docker is not running"
        print_color "YELLOW" "- Insufficient disk space"
        print_color "YELLOW" "- Port conflicts (check if ports 3000, 8000, etc. are in use)"
        exit 1
    fi
}

main() {
    print_banner
    print_color "GREEN" "AutoGPT Setup Script"
    print_color "GREEN" "-------------------"
    
    check_prerequisites
    detect_repo
    clone_repo
    if [ "$WITH_OLLAMA" = true ]; then
        bootstrap_ollama
        write_local_env
    fi
    run_docker

    echo
    print_color "GREEN" "============================="
    print_color "GREEN" "     Setup Complete!"
    print_color "GREEN" "============================="
    echo
    print_color "BLUE" "🚀 Access AutoGPT at: http://localhost:3000"
    print_color "BLUE" "📡 API available at: http://localhost:8000"
    if [ "$WITH_OLLAMA" = true ]; then
        echo
        print_color "BLUE" "🦙 AutoPilot wired to Ollama (model: $OLLAMA_MODEL)"
        print_color "YELLOW" "  Extended-thinking mode auto-downgrades to fast — Ollama"
        print_color "YELLOW" "  doesn't speak Anthropic's wire protocol. See"
        print_color "YELLOW" "  docs/platform/copilot-local-llm.md."
    fi
    echo
    print_color "YELLOW" "To stop services: docker compose down"
    print_color "YELLOW" "To view logs: docker compose logs -f"
    echo
    print_color "YELLOW" "All commands should be run in: $REPO_DIR/autogpt_platform"
}

main