# AutoGPT Platform Installers

This directory contains installation scripts to help you quickly set up the AutoGPT platform on different operating systems.

These are supposed to be run outside of the autogpt repo folder like Desktop because it will clone the repo into the current directory.

## Available Installers

- `setup-autogpt.sh` - For Linux/macOS users
- `setup-autogpt.bat` - For Windows users

## Prerequisites

Before running the installers, make sure you have the following installed:

- **Git**: For cloning the repository
- **Docker**: For running the backend services
- **Node.js and npm**: For the frontend application

## Installation Instructions

### Linux/macOS

1. Open a terminal
2. Navigate to the directory where you downloaded the installer
3. Make the script executable:
   ```bash
   chmod +x setup-autogpt.sh
   ```
4. Run the installer:
   ```bash
   ./setup-autogpt.sh
   ```

### Windows

1. Open Command Prompt or PowerShell
2. Navigate to the directory where you downloaded the installer
3. Run the installer:
   ```
   setup-autogpt.bat
   ```

## What the Installers Do

The installation scripts will:

1. Check for required prerequisites (Git, Docker, npm)
2. Clone the AutoGPT repository
3. Set up the backend services using Docker
4. Set up the frontend application
5. Start both the backend and frontend services

## After Installation

Once the installation is complete:
- The backend services will be running in Docker containers
- The frontend application will be available at http://localhost:3000

## Stopping the Services

### Linux/macOS/Windows
Press Ctrl+C in the terminal where the frontend is running, then run:
```bash
cd AutoGPT/autogpt_platform
docker compose down
```

## Troubleshooting

If you encounter any issues during installation:

1. Make sure all prerequisites are correctly installed
2. Check that Docker is running
3. Ensure you have a stable internet connection
4. Verify you have sufficient permissions to create directories and run Docker