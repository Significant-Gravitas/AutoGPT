# AutoGPT Platform Installer

The AutoGPT Platform provides easy-to-use installers to help you quickly set up the platform on your system. This page covers how to use the installer scripts for both Linux/macOS and Windows.

## What the Installer Does

The installer scripts will:

1. Check for required prerequisites (Git, Docker, npm)
2. Clone the AutoGPT repository
3. Set up the backend services using Docker
4. Set up the frontend application
5. Start both the backend and frontend services

## Prerequisites

Before running the installer, make sure you have the following installed:

- **Git**: For cloning the repository
- **Docker**: For running the backend services
- **Node.js and npm**: For the frontend application

## Quick One-Liner Installation

For convenience, you can use the following one-liner commands to install AutoGPT Platform:

### Linux/macOS

```bash
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

### Windows

```powershell
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

## Manual Installation

If you prefer, you can manually download and run the installer scripts:

- **Linux/macOS:** `setup-autogpt.sh`
- **Windows:** `setup-autogpt.bat`

These scripts are located in the `autogpt_platform/installer/` directory.

## After Installation

Once the installation is complete:
- The backend services will be running in Docker containers
- The frontend application will be available at http://localhost:3000

## Stopping the Services

To stop the services, press Ctrl+C in the terminal where the frontend is running, then run:

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