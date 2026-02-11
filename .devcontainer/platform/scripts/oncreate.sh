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
# Setup PATH for tools installed by devcontainer features
# =============================================================================
# Python feature installs pipx at /usr/local/py-utils/bin
# Node feature installs nvm, node, pnpm at various locations
export PATH="/usr/local/py-utils/bin:$PATH"

# Source nvm if available (Node feature uses nvm)
export NVM_DIR="${NVM_DIR:-/usr/local/share/nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    . "$NVM_DIR/nvm.sh"
fi

# =============================================================================
# Verify and Install Poetry
# =============================================================================
echo "📦 Setting up Poetry..."

if command -v poetry &> /dev/null; then
    echo "  Poetry already installed: $(poetry --version)"
else
    echo "  Installing Poetry via pipx..."
    if command -v pipx &> /dev/null; then
        pipx install poetry
    else
        echo "  pipx not found, installing poetry via pip..."
        pip install --user poetry
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

poetry --version || { echo "❌ Poetry installation failed"; exit 1; }

# =============================================================================
# Verify and Install pnpm
# =============================================================================
echo "📦 Setting up pnpm..."

if command -v pnpm &> /dev/null; then
    echo "  pnpm already installed: $(pnpm --version)"
else
    echo "  Installing pnpm via npm..."
    npm install -g pnpm
fi

pnpm --version || { echo "❌ pnpm installation failed"; exit 1; }

# =============================================================================
# Navigate to workspace
# =============================================================================
cd /workspaces/AutoGPT/autogpt_platform

# =============================================================================
# Install Backend Dependencies
# =============================================================================
echo "📦 Installing backend dependencies..."

cd backend
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
pnpm install --frozen-lockfile
cd ..

# =============================================================================
# Pull Dependency Docker Images
# =============================================================================
# Use docker compose pull to get exact versions from compose files
# (single source of truth, no version drift)
# =============================================================================
echo "🐳 Pulling dependency Docker images..."

# Start Docker daemon if using docker-in-docker
if [ -e /var/run/docker-host.sock ]; then
    sudo ln -sf /var/run/docker-host.sock /var/run/docker.sock || true
fi

# Check if Docker is available
if command -v docker &> /dev/null && docker info &> /dev/null; then
    # Pull images defined in docker-compose.yml (single source of truth)
    docker compose pull db redis rabbitmq kong auth || echo "⚠️ Some images may not have pulled"
    echo "✅ Dependency images pulled"
else
    echo "⚠️ Docker not available during prebuild, images will be pulled on first start"
fi

# =============================================================================
# Copy environment files
# =============================================================================
echo "📄 Setting up environment files..."

cd /workspaces/AutoGPT/autogpt_platform

[ ! -f backend/.env ] && cp backend/.env.default backend/.env
[ ! -f frontend/.env ] && cp frontend/.env.default frontend/.env
[ ! -f .env ] && cp .env.default .env

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "=============================================="
echo "✅ PREBUILD COMPLETE"
echo "=============================================="
echo ""
echo "Cached:"
echo "  ✅ Poetry $(poetry --version 2>/dev/null || echo '(check path)')"
echo "  ✅ pnpm $(pnpm --version 2>/dev/null || echo '(check path)')"
echo "  ✅ Python packages"
echo "  ✅ Node packages"
echo ""
