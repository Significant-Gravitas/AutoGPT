#!/bin/bash
# =============================================================================
# ONCREATE SCRIPT - Runs during prebuild
# =============================================================================
# This script is executed during the prebuild phase. Its job is to do all the
# heavy lifting that takes time: installing dependencies, pulling Docker images,
# generating code, etc. This makes codespace creation nearly instant.
# =============================================================================

set -e  # Exit on error
set -x  # Print commands for debugging

echo "🚀 Starting prebuild setup..."

# Workspace is autogpt_platform
cd /workspaces/AutoGPT/autogpt_platform

# =============================================================================
# Install Backend Dependencies
# =============================================================================
echo "📦 Installing backend dependencies..."

cd backend

# Install Poetry if not present
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python dependencies
poetry install --no-interaction --no-ansi

# Generate Prisma client (can be done without DB)
echo "🔧 Generating Prisma client..."
poetry run prisma generate || true
poetry run gen-prisma-stub || true

cd ..

# =============================================================================
# Install Frontend Dependencies
# =============================================================================
echo "📦 Installing frontend dependencies..."

cd frontend

# Install pnpm if not present
if ! command -v pnpm &> /dev/null; then
    echo "Installing pnpm..."
    npm install -g pnpm
fi

# Install Node dependencies
pnpm install --frozen-lockfile

# Generate API client types (if possible without backend running)
pnpm generate:api-client || echo "API client generation skipped (backend not running)"

cd ..

# =============================================================================
# Pull Docker Images (in parallel for speed)
# =============================================================================
echo "🐳 Pulling Docker images..."

# Start Docker daemon if using docker-in-docker
if [ -e /var/run/docker-host.sock ]; then
    sudo ln -sf /var/run/docker-host.sock /var/run/docker.sock || true
fi

# Pull images in parallel using background processes
docker pull supabase/gotrue:v2.170.0 &
docker pull supabase/studio:20250224-d10db0f &
docker pull kong:2.8.1 &
docker pull supabase/postgres:15.8.1.060 &
docker pull redis:latest &
docker pull rabbitmq:management &
docker pull clamav/clamav-debian:latest &

# Wait for all pulls to complete
wait

echo "✅ Docker images pulled"

# =============================================================================
# Copy environment files
# =============================================================================
echo "📄 Setting up environment files..."

cd /workspaces/AutoGPT/autogpt_platform

# Backend
if [ ! -f backend/.env ]; then
    cp backend/.env.default backend/.env
fi

# Frontend
if [ ! -f frontend/.env ]; then
    cp frontend/.env.default frontend/.env
fi

# Platform root
if [ ! -f .env ]; then
    cp .env.default .env
fi

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "=============================================="
echo "✅ PREBUILD COMPLETE"
echo "=============================================="
echo ""
echo "Dependencies installed, images pulled."
echo "The postcreate script will start services."
echo ""
