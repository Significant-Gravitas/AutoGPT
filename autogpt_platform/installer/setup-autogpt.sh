#!/bin/bash

# ------------------------------------------------------------------------------
# AutoGPT Setup Script
# ------------------------------------------------------------------------------
# This script automates the installation and setup of AutoGPT on Linux systems.
# It checks prerequisites, clones the repository, sets up backend and frontend,
# configures Sentry (optional), and starts all services. Designed for clarity
# and maintainability. Run this script from a terminal.
# ------------------------------------------------------------------------------

# --- Global Variables ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
REPO_DIR=""
CLONE_NEEDED=false
DOCKER_CMD="docker"
DOCKER_COMPOSE_CMD="docker compose"
LOG_DIR=""
SENTRY_ENABLED=0

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

# Print colored text
print_color() {
    printf "${!1}%s${NC}\n" "$2"
}

# Print the ASCII banner
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

# Handle errors and exit
handle_error() {
    echo ""
    print_color "RED" "Error: $1"
    print_color "YELLOW" "Press Enter to exit..."
    read -r
    exit 1
}

# ------------------------------------------------------------------------------
# Logging Functions
# ------------------------------------------------------------------------------

# Prepare log directory
setup_logs() {
    LOG_DIR="$REPO_DIR/autogpt_platform/logs"
    mkdir -p "$LOG_DIR"
}

# ------------------------------------------------------------------------------
# Health Check Functions
# ------------------------------------------------------------------------------

# Check service health by polling an endpoint
check_health() {
    local url=$1
    local expected=$2
    local name=$3
    local max_attempts=$4
    local timeout=$5

    if ! command -v curl &> /dev/null; then
        echo "curl not found. Skipping health check for $name."
        return 0
    fi

    echo "Checking $name health..."
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        echo "Attempt $attempt/$max_attempts"
        response=$(curl -s --max-time "$timeout" "$url")
        if [[ "$response" == *"$expected"* ]]; then
            echo "✓ $name is healthy"
            return 0
        fi
        echo "Waiting 5s before next attempt..."
        sleep 5
    done
    echo "✗ $name health check failed after $max_attempts attempts"
    return 1
}

# ------------------------------------------------------------------------------
# Prerequisite and Environment Functions
# ------------------------------------------------------------------------------

# Check for required commands
check_command() {
    local cmd=$1
    local name=$2
    local url=$3

    if ! command -v "$cmd" &> /dev/null; then
        handle_error "$name is not installed. Please install it and try again. Visit $url"
    else
        print_color "GREEN" "✓ $name is installed"
    fi
}

# Check for optional commands
check_command_optional() {
    local cmd=$1
    if command -v "$cmd" &> /dev/null; then
        print_color "GREEN" "✓ $cmd is installed"
    else
        print_color "YELLOW" "$cmd is not installed. Some features will be skipped."
    fi
}

# Check Docker permissions and adjust commands if needed
check_docker_permissions() {
    if ! docker info &> /dev/null; then
        print_color "YELLOW" "Docker requires elevated privileges. Using sudo for Docker commands..."
        DOCKER_CMD="sudo docker"
        DOCKER_COMPOSE_CMD="sudo docker compose"
    fi
}

# Check all prerequisites
check_prerequisites() {
    print_color "GREEN" "AutoGPT's Automated Setup Script"
    print_color "GREEN" "-------------------------------"
    print_color "BLUE" "This script will automatically install and set up AutoGPT for you."
    echo ""
    print_color "YELLOW" "Checking prerequisites:"

    check_command git "Git" "https://git-scm.com/downloads"
    check_command docker "Docker" "https://docs.docker.com/get-docker/"
    check_docker_permissions
    check_command npm "npm (Node.js)" "https://nodejs.org/en/download/"
    check_command pnpm "pnpm (Node.js package manager)" "https://pnpm.io/installation"
    check_command_optional curl "curl"

    print_color "GREEN" "All prerequisites are installed! Starting installation..."
    echo ""
}

# Detect installation mode and set repo directory
# (Clones if not in a repo, otherwise uses current directory)
detect_installation_mode() {
    if [[ "$PWD" == */autogpt_platform/installer ]]; then
        if [[ -d "../../.git" ]]; then
            REPO_DIR="$(cd ../..; pwd)"
            CLONE_NEEDED=false
            cd ../.. || handle_error "Failed to navigate to repository root."
        else
            CLONE_NEEDED=true
            REPO_DIR="$(pwd)/AutoGPT"
            cd "$(dirname \"$(dirname \"$(dirname \"$PWD\")\")\")" || handle_error "Failed to navigate to parent directory."
        fi
    elif [[ -d ".git" && -d "autogpt_platform/installer" ]]; then
        REPO_DIR="$PWD"
        CLONE_NEEDED=false
    else
        CLONE_NEEDED=true
        REPO_DIR="$(pwd)/AutoGPT"
    fi
}

# Clone the repository if needed
clone_repository() {
    if [ "$CLONE_NEEDED" = true ]; then
        print_color "BLUE" "Cloning AutoGPT repository..."
        if git clone https://github.com/Significant-Gravitas/AutoGPT.git "$REPO_DIR"; then
            print_color "GREEN" "✓ Repo cloned successfully!"
        else
            handle_error "Failed to clone the repository."
        fi
    else
        print_color "GREEN" "Using existing AutoGPT repository"
    fi
}

# Prompt for Sentry enablement and set global flag
prompt_sentry_enablement() {
    print_color "YELLOW" "Would you like to enable debug information to be shared so we can fix your issues? [Y/n]"
    read -r sentry_answer
    case "${sentry_answer,,}" in
        ""|y|yes)
            SENTRY_ENABLED=1
            ;;
        n|no)
            SENTRY_ENABLED=0
            ;;
        *)
            print_color "YELLOW" "Invalid input. Defaulting to yes. Sentry will be enabled."
            SENTRY_ENABLED=1
            ;;
    esac
}

# ------------------------------------------------------------------------------
# Setup Functions
# ------------------------------------------------------------------------------

# Set up backend services and configure Sentry if enabled
setup_backend() {
    print_color "BLUE" "Setting up backend services..."
    cd "$REPO_DIR/autogpt_platform" || handle_error "Failed to navigate to backend directory."
    cp .env.example .env || handle_error "Failed to copy environment file."

    # Set SENTRY_DSN in backend/.env
    cd backend || handle_error "Failed to navigate to backend subdirectory."
    cp .env.example .env || handle_error "Failed to copy backend environment file."
    sentry_url="https://11d0640fef35640e0eb9f022eb7d7626@o4505260022104064.ingest.us.sentry.io/4507890252447744"
    if [ "$SENTRY_ENABLED" = "1" ]; then
        sed -i "s|^SENTRY_DSN=.*$|SENTRY_DSN=$sentry_url|" .env || echo "SENTRY_DSN=$sentry_url" >> .env
        print_color "GREEN" "Sentry enabled in backend."
    else
        sed -i "s|^SENTRY_DSN=.*$|SENTRY_DSN=|" .env || echo "SENTRY_DSN=" >> .env
        print_color "YELLOW" "Sentry not enabled in backend."
    fi
    cd .. # back to autogpt_platform

    $DOCKER_COMPOSE_CMD down || handle_error "Failed to stop existing backend services."
    $DOCKER_COMPOSE_CMD up -d --build || handle_error "Failed to start backend services."
    print_color "GREEN" "✓ Backend services started successfully"
}

# Set up frontend application
setup_frontend() {
    print_color "BLUE" "Setting up frontend application..."
    cd "$REPO_DIR/autogpt_platform/frontend" || handle_error "Failed to navigate to frontend directory."
    cp .env.example .env || handle_error "Failed to copy frontend environment file."
    corepack enable || handle_error "Failed to enable corepack."
    pnpm install || handle_error "Failed to install frontend dependencies."
    print_color "GREEN" "✓ Frontend dependencies installed successfully"
}

# Run backend and frontend setup concurrently and manage logs
run_concurrent_setup() {
    setup_logs
    backend_log="$LOG_DIR/backend_setup.log"
    frontend_log="$LOG_DIR/frontend_setup.log"

    : > "$backend_log"
    : > "$frontend_log"

    setup_backend > "$backend_log" 2>&1 &
    backend_pid=$!
    echo "Backend setup finished."

    setup_frontend > "$frontend_log" 2>&1 &
    frontend_pid=$!
    echo "Frontend setup finished."

    show_spinner "$backend_pid" "$frontend_pid"

    wait $backend_pid; backend_status=$?
    wait $frontend_pid; frontend_status=$?

    if [ $backend_status -ne 0 ]; then
        print_color "RED" "Backend setup failed. See log: $backend_log"
        exit 1
    fi

    if [ $frontend_status -ne 0 ]; then
        print_color "RED" "Frontend setup failed. See log: $frontend_log"
        exit 1
    fi

}

# Show a spinner while background jobs run
show_spinner() {
    local backend_pid=$1
    local frontend_pid=$2
    spin='-\|/'
    i=0
    messages=("Working..." "Still working..." "Setting up dependencies..." "Almost there...")
    msg_index=0
    msg_counter=0
    clear_line="                                                                               "

    while kill -0 $backend_pid 2>/dev/null || kill -0 $frontend_pid 2>/dev/null; do
        i=$(( (i+1) % 4 ))
        msg_counter=$(( (msg_counter+1) % 300 ))
        if [ $msg_counter -eq 0 ]; then
            msg_index=$(( (msg_index+1) % ${#messages[@]} ))
        fi
        printf "\r${clear_line}\r${YELLOW}[%c]${NC} %s" "${spin:$i:1}" "${messages[$msg_index]}"
        sleep .1
    done
    printf "\r${clear_line}\r${GREEN}[✓]${NC} Setup completed!\n"
}

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

main() {
    print_banner
    check_prerequisites
    prompt_sentry_enablement
    detect_installation_mode
    clone_repository
    setup_logs
    run_concurrent_setup

    print_color "YELLOW" "Starting frontend..."
    (cd "$REPO_DIR/autogpt_platform/frontend" && pnpm dev > "$LOG_DIR/frontend_dev.log" 2>&1 &)

    print_color "YELLOW" "Waiting for services to start..."
    sleep 20

    print_color "YELLOW" "Verifying services health..."
    check_health "http://localhost:8006/health" "\"status\":\"healthy\"" "Backend" 6 15 
    check_health "http://localhost:3000/health" "Yay im healthy" "Frontend" 6 15

    if [ $backend_status -ne 0 ] || [ $frontend_status -ne 0 ]; then
        print_color "RED" "Setup failed. See logs for details."
        exit 1
    fi

    print_color "GREEN" "Setup complete!"
    print_color "BLUE" "Access AutoGPT at: http://localhost:3000"
    print_color "YELLOW" "To stop services, press Ctrl+C and run 'docker compose down' in $REPO_DIR/autogpt_platform"
    echo ""
    print_color "GREEN" "Press Enter to exit (services will keep running)..."
    read -r
}

main
