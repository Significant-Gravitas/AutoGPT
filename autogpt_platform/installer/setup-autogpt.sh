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
NC='\033[0m'

# Variables
REPO_DIR=""
CLONE_NEEDED=false
DOCKER_CMD="docker"
DOCKER_COMPOSE_CMD="docker compose"
SENTRY_ENABLED=0
LOG_FILE=""

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

check_command() {
    local cmd=$1
    local name=$2
    
    if ! command -v "$cmd" &> /dev/null; then
        handle_error "$name is not installed. Please install it and try again."
    else
        print_color "GREEN" "✓ $name is installed"
    fi
}

check_prerequisites() {
    print_color "BLUE" "Checking prerequisites..."
    check_command git "Git"
    check_command docker "Docker"
    
    if ! docker info &> /dev/null; then
        print_color "YELLOW" "Using sudo for Docker commands..."
        DOCKER_CMD="sudo docker"
        DOCKER_COMPOSE_CMD="sudo docker compose"
    fi
    
    print_color "GREEN" "All prerequisites installed!"
}

prompt_sentry() {
    print_color "YELLOW" "Enable debug info sharing to help fix issues? [Y/n]"
    read -r sentry_answer
    case "${sentry_answer,,}" in
        ""|y|yes) SENTRY_ENABLED=1 ;;
        *) SENTRY_ENABLED=0 ;;
    esac
}

detect_repo() {
    if [[ "$PWD" == */autogpt_platform/installer ]]; then
        if [[ -d "../../.git" ]]; then
            REPO_DIR="$(cd ../..; pwd)"
            cd ../.. || handle_error "Failed to navigate to repo root"
        else
            CLONE_NEEDED=true
            REPO_DIR="$(pwd)/AutoGPT"
        fi
    elif [[ -d ".git" && -d "autogpt_platform/installer" ]]; then
        REPO_DIR="$PWD"
    else
        CLONE_NEEDED=true
        REPO_DIR="$(pwd)/AutoGPT"
    fi
}

clone_repo() {
    if [ "$CLONE_NEEDED" = true ]; then
        print_color "BLUE" "Cloning AutoGPT repository..."
        git clone https://github.com/Significant-Gravitas/AutoGPT.git "$REPO_DIR" || handle_error "Failed to clone repository"
    fi
}

setup_env() {
    cd "$REPO_DIR/autogpt_platform" || handle_error "Failed to navigate to autogpt_platform"
    
    # Copy main .env
    cp .env.example .env || handle_error "Failed to copy main .env"
    
    # Configure backend Sentry
    cd backend || handle_error "Failed to navigate to backend"
    cp .env.example .env || handle_error "Failed to copy backend .env"
    
    local sentry_url="https://11d0640fef35640e0eb9f022eb7d7626@o4505260022104064.ingest.us.sentry.io/4507890252447744"
    if [ "$SENTRY_ENABLED" = "1" ]; then
        sed -i "s|^SENTRY_DSN=.*$|SENTRY_DSN=$sentry_url|" .env || echo "SENTRY_DSN=$sentry_url" >> .env
        print_color "GREEN" "Sentry enabled"
    else
        sed -i "s|^SENTRY_DSN=.*$|SENTRY_DSN=|" .env || echo "SENTRY_DSN=" >> .env
    fi
    
    cd ..
}

run_docker() {
    print_color "BLUE" "Running docker compose up -d --build..."
    mkdir -p logs
    LOG_FILE="$REPO_DIR/autogpt_platform/logs/docker_setup.log"
    
    if $DOCKER_COMPOSE_CMD up -d --build > "$LOG_FILE" 2>&1; then
        print_color "GREEN" "✓ Services started successfully"
    else
        handle_error "Docker compose failed"
    fi
}

main() {
    print_banner
    print_color "GREEN" "AutoGPT Setup Script"
    print_color "GREEN" "-------------------"
    
    check_prerequisites
    prompt_sentry
    detect_repo
    clone_repo
    setup_env
    run_docker
    
    print_color "GREEN" "Setup complete!"
    print_color "BLUE" "Access AutoGPT at: http://localhost:3000"
    print_color "YELLOW" "To stop: 'docker compose down' in $REPO_DIR/autogpt_platform"
}

main