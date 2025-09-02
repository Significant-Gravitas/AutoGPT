#!/bin/bash

# AutoGPT Local Access via Port Forwarding
# This script sets up port forwarding for local access to all AutoGPT services

echo "🚀 Starting AutoGPT local access via port forwarding..."
echo "⚠️  Keep this terminal open while using AutoGPT"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all port forwards..."
    jobs -p | xargs -r kill
    exit 0
}

# Trap cleanup on script exit
trap cleanup SIGINT SIGTERM

# Backend API Server (port 8006)
echo "📡 Starting API server port forward (localhost:8006)..."
kubectl port-forward deployment/autogpt-server 8006:8006 -n autogpt &
API_PID=$!

# Frontend Builder (port 3000)  
echo "🖥️  Starting frontend port forward (localhost:3000)..."
kubectl port-forward deployment/autogpt-builder 3000:3000 -n autogpt &
FRONTEND_PID=$!

# Supabase Auth (port 9999)
echo "🔐 Starting auth server port forward (localhost:9999)..."
kubectl port-forward deployment/supabase-auth 9999:9999 -n autogpt &
AUTH_PID=$!

# Websocket Server (port 8001) - if deployed
echo "🔌 Starting websocket server port forward (localhost:8001)..."
kubectl port-forward deployment/autogpt-websocket-server 8001:8001 -n autogpt &
WS_PID=$!

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are responding
echo ""
echo "🔍 Testing service health..."

# Test API
if curl -s http://localhost:8006/health >/dev/null 2>&1; then
    echo "✅ API Server: http://localhost:8006"
else
    echo "❌ API Server: Failed to connect"
fi

# Test Frontend
if curl -s http://localhost:3000/ >/dev/null 2>&1; then
    echo "✅ Frontend: http://localhost:3000"
else
    echo "❌ Frontend: Failed to connect"
fi

# Test Auth
if curl -s http://localhost:9999/health >/dev/null 2>&1; then
    echo "✅ Auth Server: http://localhost:9999"
else
    echo "❌ Auth Server: Failed to connect"
fi

echo ""
echo "🎉 AutoGPT is ready!"
echo ""
echo "📋 Access URLs:"
echo "   Frontend:    http://localhost:3000"
echo "   API:         http://localhost:8006"
echo "   Auth:        http://localhost:9999"
echo "   Websockets:  http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop all port forwards"

# Wait for user to stop
wait