#!/bin/bash
# =============================================================================
# POSTCREATE SCRIPT - Runs after container creation
# =============================================================================
# This script runs once when a codespace is first created. It starts the
# dependency services and prepares the environment for development.
#
# NOTE: Backend and Frontend run NATIVELY (not in Docker) to ensure you're
# always running the current branch's code, not stale prebuild code.
# =============================================================================

set -e  # Exit on error

echo "🚀 Setting up your development environment..."

cd /workspaces/AutoGPT/autogpt_platform

# =============================================================================
# Ensure Docker is available
# =============================================================================
if [ -e /var/run/docker-host.sock ]; then
    sudo ln -sf /var/run/docker-host.sock /var/run/docker.sock 2>/dev/null || true
fi

# Wait for Docker to be ready
echo "⏳ Waiting for Docker..."
timeout 60 bash -c 'until docker info &>/dev/null; do sleep 1; done'
echo "✅ Docker is ready"

# =============================================================================
# Start Dependency Services ONLY
# =============================================================================
# We only start infrastructure deps in Docker.
# Backend/Frontend run natively to use the current branch's code.
# =============================================================================
echo "🐳 Starting dependency services..."

# Start core dependencies (DB, Auth, Redis, RabbitMQ)
docker compose up -d db redis rabbitmq kong auth

# Wait for PostgreSQL to be healthy
echo "⏳ Waiting for PostgreSQL..."
timeout 120 bash -c '
until docker compose exec -T db pg_isready -U postgres &>/dev/null; do
    sleep 2
    echo "  Waiting for database..."
done
'
echo "✅ PostgreSQL is ready"

# Wait for Redis
echo "⏳ Waiting for Redis..."
timeout 60 bash -c 'until docker compose exec -T redis redis-cli ping &>/dev/null; do sleep 1; done'
echo "✅ Redis is ready"

# Wait for RabbitMQ
echo "⏳ Waiting for RabbitMQ..."
timeout 90 bash -c 'until docker compose exec -T rabbitmq rabbitmq-diagnostics -q ping &>/dev/null; do sleep 2; done'
echo "✅ RabbitMQ is ready"

# =============================================================================
# Run Database Migrations
# =============================================================================
echo "🔄 Running database migrations..."

cd backend

# Ensure Poetry is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Run migrations
poetry run prisma migrate deploy
poetry run prisma generate
poetry run gen-prisma-stub || true

cd ..

# =============================================================================
# Seed Test Data (Minimal)
# =============================================================================
echo "🌱 Checking test data..."

cd backend

# Check if test data already exists (idempotent)
if poetry run python -c "
import asyncio
from backend.data.db import prisma

async def check():
    await prisma.connect()
    count = await prisma.user.count()
    await prisma.disconnect()
    return count > 0

print('exists' if asyncio.run(check()) else 'empty')
" 2>/dev/null | grep -q "exists"; then
    echo "  Test data already exists, skipping seed"
else
    echo "  Running E2E test data creator..."
    poetry run python test/e2e_test_data.py || echo "⚠️ Test data seeding had issues (may be partial)"
fi

cd ..

# =============================================================================
# Print Welcome Message
# =============================================================================
echo ""
echo "=============================================="
echo "🎉 CODESPACE READY!"
echo "=============================================="
echo ""
echo "📍 Services Running (Docker):"
echo "   PostgreSQL:  localhost:5432"
echo "   Redis:       localhost:6379"
echo "   RabbitMQ:    localhost:5672 (mgmt: 15672)"
echo "   Supabase:    localhost:8000"
echo ""
echo "🚀 Start Development:"
echo "   make run-backend   # Start backend (localhost:8006)"
echo "   make run-frontend  # Start frontend (localhost:3000)"
echo ""
echo "   Or run both in separate terminals!"
echo ""
echo "🔑 Test Account:"
echo "   Email:    test123@gmail.com"
echo "   Password: testpassword123"
echo ""
echo "📚 Full docs: .devcontainer/README.md"
echo ""
