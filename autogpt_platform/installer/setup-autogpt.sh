#!/bin/bash

# ------------------------------------------------------------------------------
# AutoGPT Setup Script
# ------------------------------------------------------------------------------
# This script automates the installation and setup of AutoGPT on Linux systems.
# It checks prerequisites, clones the repository, and starts all services.
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
    run_docker
    
    echo
    print_color "GREEN" "============================="
    print_color "GREEN" "     Setup Complete!"
    print_color "GREEN" "============================="
    echo
    print_color "BLUE" "🚀 Access AutoGPT at: http://localhost:3000"
    print_color "BLUE" "📡 API available at: http://localhost:8000"
    echo
    print_color "YELLOW" "To stop services: docker compose down"
    print_color "YELLOW" "To view logs: docker compose logs -f"
    echo
    print_color "YELLOW" "All commands should be run in: $REPO_DIR/autogpt_platform"
}

main