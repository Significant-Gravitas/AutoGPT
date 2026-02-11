#!/bin/bash
# =============================================================================
# POSTCREATE SCRIPT - Runs after container creation
# =============================================================================
# This script runs once when a codespace is first created. It starts the
# services, runs migrations, and seeds the database with test data.
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
# Start Core Services
# =============================================================================
echo "🐳 Starting Docker services..."

# Start services (Supabase DB, Auth, Kong, Redis, RabbitMQ)
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

# Wait for other services
echo "⏳ Waiting for Redis..."
timeout 60 bash -c 'until docker compose exec -T redis redis-cli ping &>/dev/null; do sleep 1; done'
echo "✅ Redis is ready"

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
# Seed Test Data
# =============================================================================
echo "🌱 Seeding test data..."

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
# Start Backend Services (optional - for immediate use)
# =============================================================================
echo "🚀 Starting backend services..."

# Start remaining services
docker compose up -d migrate
sleep 5  # Wait for migration container

docker compose up -d rest_server executor websocket_server database_manager

# =============================================================================
# Print Welcome Message
# =============================================================================
echo ""
echo "=============================================="
echo "🎉 CODESPACE READY!"
echo "=============================================="
echo ""
echo "📍 Quick Links (will auto-forward):"
echo "   Frontend:  http://localhost:3000"
echo "   API:       http://localhost:8006"
echo "   Supabase:  http://localhost:8000"
echo ""
echo "🔑 Test Account:"
echo "   Email:    test123@gmail.com"
echo "   Password: testpassword123"
echo ""
echo "🛠️  Development Commands:"
echo "   make run-frontend  # Start frontend dev server"
echo "   make run-backend   # Start backend dev server"
echo "   make start-core    # Start only core services"
echo "   make test-data     # Regenerate test data"
echo ""
echo "📚 See README.md for more info"
echo ""
