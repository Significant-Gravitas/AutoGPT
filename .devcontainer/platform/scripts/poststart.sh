#!/bin/bash
# =============================================================================
# POSTSTART SCRIPT - Runs every time the codespace starts
# =============================================================================
# This script runs when:
# 1. Codespace is first created (after postcreate)
# 2. Codespace resumes from stopped state
# 3. Codespace rebuilds
#
# It ensures dependency services are running. Backend/Frontend are run
# manually by the developer for hot-reload during development.
# =============================================================================

echo "🔄 Starting dependency services..."

cd /workspaces/AutoGPT/autogpt_platform

# Ensure Docker socket is available
if [ -e /var/run/docker-host.sock ]; then
    sudo ln -sf /var/run/docker-host.sock /var/run/docker.sock 2>/dev/null || true
fi

# Wait for Docker
timeout 30 bash -c 'until docker info &>/dev/null; do sleep 1; done' || {
    echo "⚠️ Docker not available, services may need manual start"
    exit 0
}

# Start only dependency services (not backend/frontend)
docker compose up -d db redis rabbitmq kong auth

# Quick health check
echo "⏳ Waiting for services..."
sleep 5

if docker compose ps | grep -q "running"; then
    echo "✅ Dependency services are running"
    echo ""
    echo "🚀 Start development with:"
    echo "   make run-backend   # Terminal 1"
    echo "   make run-frontend  # Terminal 2"
else
    echo "⚠️ Some services may not be running. Try: docker compose up -d"
fi
