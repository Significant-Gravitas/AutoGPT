#!/bin/bash

# AutoGPT Setup Script
# Works on Linux

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Globals
REPO_DIR=""
CLONE_NEEDED=false
DOCKER_CMD="docker"
DOCKER_COMPOSE_CMD="docker compose"
LOG_DIR=""

# ------------------ Helper Functions ------------------

print_color() {
    printf "${!1}%s${NC}\n" "$2"
}

# Function to print the banner
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

# Function to handle errors
handle_error() {
    echo ""
    print_color "RED" "Error: $1"
    print_color "YELLOW" "Press Enter to exit..."
    read -r
    exit 1
}

# Function to check prerequisites
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
    check_command_optional curl "curl"

    print_color "GREEN" "All prerequisites are installed! Starting installation..."
    echo ""
}

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

check_command_optional() {
    local cmd=$1
    if command -v "$cmd" &> /dev/null; then
        print_color "GREEN" "✓ $cmd is installed"
    else
        print_color "YELLOW" "$cmd is not installed. Some features will be skipped."
    fi
}

check_docker_permissions() {
    if ! docker info &> /dev/null; then
        print_color "YELLOW" "Docker requires elevated privileges. Using sudo for Docker commands..."
        DOCKER_CMD="sudo docker"
        DOCKER_COMPOSE_CMD="sudo docker compose"
    fi
}

# ------------------ Setup ------------------

detect_installation_mode() {
    if [[ "$PWD" == */autogpt_platform/installer ]]; then
        if [[ -d "../../.git" ]]; then
            REPO_DIR="$(cd ../..; pwd)"
            CLONE_NEEDED=false
            cd ../.. || handle_error "Failed to navigate to repository root."
        else
            CLONE_NEEDED=true
            REPO_DIR="$(pwd)/AutoGPT"
            cd "$(dirname "$(dirname "$(dirname "$PWD")")")" || handle_error "Failed to navigate to parent directory."
        fi
    elif [[ -d ".git" && -d "autogpt_platform/installer" ]]; then
        REPO_DIR="$PWD"
        CLONE_NEEDED=false
    else
        CLONE_NEEDED=true
        REPO_DIR="$(pwd)/AutoGPT"
    fi
}

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


setup_backend() {
    print_color "BLUE" "Setting up backend services..."
    cd "$REPO_DIR/autogpt_platform" || handle_error "Failed to navigate to backend directory."
    cp .env.example .env || handle_error "Failed to copy environment file."
    $DOCKER_COMPOSE_CMD up -d --build || handle_error "Failed to start backend services."
    print_color "GREEN" "✓ Backend services started successfully"
}

setup_frontend() {
    print_color "BLUE" "Setting up frontend application..."
    cd "$REPO_DIR/autogpt_platform/frontend" || handle_error "Failed to navigate to frontend directory."
    cp .env.example .env || handle_error "Failed to copy frontend environment file."
    npm install || handle_error "Failed to install frontend dependencies."
    print_color "GREEN" "✓ Frontend dependencies installed successfully"
}

# ------------------ Health Check ------------------

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

# ------------------ Logging ------------------

setup_logs() {
    LOG_DIR="$REPO_DIR/autogpt_platform/logs"
    mkdir -p "$LOG_DIR"
}

save_logs() {
    cp "$1" "$LOG_DIR/backend_setup.log"
    cp "$2" "$LOG_DIR/frontend_setup.log"
}

run_concurrent_setup() {
    backend_log=$(mktemp)
    frontend_log=$(mktemp)

    setup_backend > "$backend_log" 2>&1 &
    backend_pid=$!
    echo "Backend setup started."

    setup_frontend > "$frontend_log" 2>&1 &
    frontend_pid=$!
    echo "Frontend setup started."

    show_spinner "$backend_pid" "$frontend_pid"

    wait $backend_pid; backend_status=$?
    wait $frontend_pid; frontend_status=$?

    save_logs "$backend_log" "$frontend_log"

    if [ $backend_status -ne 0 ]; then
        cat "$backend_log"
        handle_error "Backend setup failed. Logs saved to $LOG_DIR/backend_setup.log"
    fi

    if [ $frontend_status -ne 0 ]; then
        cat "$frontend_log"
        handle_error "Frontend setup failed. Logs saved to $LOG_DIR/frontend_setup.log"
    fi

    rm -f "$backend_log" "$frontend_log"
}

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

# ------------------ Main Script ------------------

main() {
    print_banner
    check_prerequisites
    detect_installation_mode
    clone_repository
    setup_logs
    run_concurrent_setup

    print_color "BLUE" "Starting frontend development server in a new terminal..."
    cd "$REPO_DIR/autogpt_platform/frontend" || handle_error "Failed to navigate to frontend directory."

    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "npm run dev; exec bash"
    else
        print_color "YELLOW" "gnome-terminal not found. Running frontend in background..."
        npm run dev &
    fi


    print_color "YELLOW" "Waiting for services to start..."
    sleep 20

    print_color "YELLOW" "Verifying service health..."
    check_health "http://localhost:8006/health" "\"status\":\"healthy\"" "Backend" 6 15 
    check_health "http://localhost:3000/health" "Yay im healthy" "Frontend" 6 15

    print_color "GREEN" "Setup complete!"
    print_color "BLUE" "Access AutoGPT at: http://localhost:3000"
    print_color "YELLOW" "To stop services, press Ctrl+C and run 'docker compose down' in $REPO_DIR/autogpt_platform"
    echo ""
    print_color "GREEN" "Press Enter to exit (services will keep running)..."
    read -r
}

main
