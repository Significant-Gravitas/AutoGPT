#!/bin/bash

# AutoGPT Setup Script
# Works on Linux

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored text
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
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        handle_error "Git is not installed. Please install Git and try again. Visit https://git-scm.com/downloads for installation instructions."
    else
        print_color "GREEN" "✓ Git is installed"
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        handle_error "Docker is not installed. Please install Docker and try again. Visit https://docs.docker.com/get-docker/ for installation instructions."
    else
        print_color "GREEN" "✓ Docker is installed"
    fi
    
    # Check if user is in docker group or has sudo access
    if ! docker info &> /dev/null; then
        print_color "YELLOW" "Docker requires elevated privileges. Using sudo for Docker commands..."
        DOCKER_CMD="sudo docker"
        DOCKER_COMPOSE_CMD="sudo docker compose"
    else
        DOCKER_CMD="docker"
        DOCKER_COMPOSE_CMD="docker compose"
    fi
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        handle_error "npm is not installed. Please install Node.js and npm and try again. Visit https://nodejs.org/en/download/ for installation instructions."
    else
        print_color "GREEN" "✓ npm is installed"
    fi
    
    print_color "GREEN" "All prerequisites are installed! Starting installation..."
    echo ""
}

# Function to run backend setup
setup_backend() {
    print_color "BLUE" "Setting up backend services..."
    cd AutoGPT/autogpt_platform || handle_error "Failed to navigate to AutoGPT/autogpt_platform directory."
    
    # Copy the example environment file
    cp .env.example .env || handle_error "Failed to copy environment file. Please check permissions and try again."
    
    # Run docker compose
    print_color "BLUE" "Starting backend services with Docker..."
    $DOCKER_COMPOSE_CMD up -d --build || handle_error "Failed to start the backend services. Please check Docker and try again."
    
    print_color "GREEN" "✓ Backend services started successfully"
    cd ..
}

# Function to run frontend setup
setup_frontend() {
    print_color "BLUE" "Setting up frontend application..."
    cd AutoGPT/autogpt_platform/frontend || handle_error "Failed to navigate to frontend directory."
    
    # Copy the frontend example environment file
    cp .env.example .env || handle_error "Failed to copy frontend environment file. Please check permissions and try again."
    
    # Install dependencies
    print_color "BLUE" "Installing frontend dependencies..."
    npm install || handle_error "Failed to install frontend dependencies. Please check npm and try again."
    
    print_color "GREEN" "✓ Frontend dependencies installed successfully"
}

# Main execution
print_banner
check_prerequisites

# Clone the repository
print_color "BLUE" "Cloning the AutoGPT repository..."
git clone https://github.com/Significant-Gravitas/AutoGPT.git || handle_error "Failed to clone the repository. Please check your internet connection and try again."

# Run backend and frontend setup concurrently
print_color "YELLOW" "Running backend and frontend setup!"

# Create a temporary file to capture output
backend_log=$(mktemp)
frontend_log=$(mktemp)

# Run backend setup in background
setup_backend > "$backend_log" 2>&1 &
backend_pid=$!

# Run frontend setup in background
setup_frontend > "$frontend_log" 2>&1 &
frontend_pid=$!

# Show a spinner while waiting for both processes
print_color "BLUE" "Setting up components (this may take a few minutes)..."
spin='-\|/'
i=0
messages=("Working..." "Still working..." "Setting up dependencies..." "This may take a few minutes..." "Almost there...")
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

# Check if any process failed
wait $backend_pid
backend_status=$?
wait $frontend_pid
frontend_status=$?

if [ $backend_status -ne 0 ]; then
    print_color "RED" "Backend setup failed. See log for details:"
    cat "$backend_log"
    handle_error "Backend setup failed"
fi

if [ $frontend_status -ne 0 ]; then
    print_color "RED" "Frontend setup failed. See log for details:"
    cat "$frontend_log"
    handle_error "Frontend setup failed"
fi

# Clean up temp files
rm -f "$backend_log" "$frontend_log"

# Start the frontend development server
print_color "BLUE" "Starting frontend development server..."
cd AutoGPT/autogpt_platform/frontend || handle_error "Failed to navigate to frontend directory."
npm run dev

print_color "GREEN" "AutoGPT setup completed successfully!"
print_color "GREEN" "-------------------------------------"
print_color "BLUE" "Your backend services are running in Docker."
print_color "BLUE" "Your frontend application is running at http://localhost:3000"
echo ""
print_color "YELLOW" "Visit http://localhost:3000 in your browser to access AutoGPT."
echo ""
print_color "YELLOW" "To stop the services, press Ctrl+C in this terminal, then run 'docker compose down' in the AutoGPT/autogpt_platform directory."
echo ""
print_color "GREEN" "Press Enter to exit the script (this will NOT stop the services)..."
read -r