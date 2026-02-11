#!/bin/bash
# =============================================================================
# ONCREATE SCRIPT - Runs during prebuild
# =============================================================================
# This script runs during the prebuild phase (GitHub Actions).
# It caches everything that's safe to cache:
#   ✅ Dependency Docker images (postgres, redis, rabbitmq, etc.)
#   ✅ Python packages (poetry install)
#   ✅ Node packages (pnpm install)
#
# It does NOT build backend/frontend Docker images because those would
# contain stale code from the prebuild branch, not the PR being reviewed.
# =============================================================================

set -e  # Exit on error
set -x  # Print commands for debugging

echo "🚀 Starting prebuild setup..."

# =============================================================================
# Install Poetry via pipx (pipx is installed by the Python devcontainer feature)
# =============================================================================
echo "📦 Installing Poetry..."

# pipx is installed by the Python feature at /usr/local/py-utils/bin
export PATH="/usr/local/py-utils/bin:$PATH"

if ! command -v poetry &> /dev/null; then
    pipx install poetry
fi

# Verify poetry is available
poetry --version

# Workspace is autogpt_platform
cd /workspaces/AutoGPT/autogpt_platform

# =============================================================================
# Install Backend Dependencies
# =============================================================================
echo "📦 Installing backend dependencies..."

cd backend

# Install Python dependencies
poetry install --no-interaction --no-ansi

# Generate Prisma client (schema only, no DB needed)
echo "🔧 Generating Prisma client..."
poetry run prisma generate || true
poetry run gen-prisma-stub || true

cd ..

# =============================================================================
# Install Frontend Dependencies
# =============================================================================
echo "📦 Installing frontend dependencies..."

cd frontend

# pnpm should be installed by the Node feature, but ensure it's available
if ! command -v pnpm &> /dev/null; then
    echo "Installing pnpm..."
    npm install -g pnpm
fi

# Install Node dependencies
pnpm install --frozen-lockfile

cd ..

# =============================================================================
# Pull Dependency Docker Images ONLY
# =============================================================================
# We only pull infrastructure images. Backend/Frontend run natively
# to ensure we always use the current branch's code.
# =============================================================================
echo "🐳 Pulling dependency Docker images..."

# Start Docker daemon if using docker-in-docker
if [ -e /var/run/docker-host.sock ]; then
    sudo ln -sf /var/run/docker-host.sock /var/run/docker.sock || true
fi

# Pull dependency images in parallel
docker pull supabase/gotrue:v2.170.0 &
docker pull supabase/studio:20250224-d10db0f &
docker pull kong:2.8.1 &
docker pull supabase/postgres:15.8.1.060 &
docker pull redis:latest &
docker pull rabbitmq:management &

# Wait for all pulls to complete
wait

echo "✅ Dependency images pulled"

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
echo "Cached:"
echo "  ✅ Poetry (via pipx)"
echo "  ✅ Python packages"
echo "  ✅ Node packages (pnpm)"
echo "  ✅ Dependency Docker images"
echo ""
echo "The postcreate script will start services."
echo ""
