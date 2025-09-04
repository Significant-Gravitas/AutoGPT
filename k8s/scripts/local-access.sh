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

# Frontend Builder (port 3000)  
echo "🖥️  Starting frontend port forward (localhost:3000)..."
kubectl port-forward svc/autogpt-builder 3000:3000 -n autogpt &
FRONTEND_PID=$!

# Backend API Server (port 8006)
echo "📡 Starting API server port forward (localhost:8006)..."
kubectl port-forward svc/autogpt-server 8006:8006 -n autogpt &
API_PID=$!

# Executor Service (port 8002)
echo "⚙️  Starting executor port forward (localhost:8002)..."
kubectl port-forward svc/autogpt-server-executor 8002:8002 -n autogpt &
EXECUTOR_PID=$!

# Websocket Server (port 8001)
echo "🔌 Starting websocket server port forward (localhost:8001)..."
kubectl port-forward svc/autogpt-websocket 8001:8001 -n autogpt &
WS_PID=$!

# Scheduler Service (port 8003)
echo "📅 Starting scheduler port forward (localhost:8003)..."
kubectl port-forward svc/autogpt-scheduler 8003:8003 -n autogpt &
SCHEDULER_PID=$!

# Database Manager (port 8005)
echo "💾 Starting database manager port forward (localhost:8005)..."
kubectl port-forward svc/autogpt-database-manager 8005:8005 -n autogpt &
DBM_PID=$!

# Notification Service (port 8007)
echo "🔔 Starting notification service port forward (localhost:8007)..."
kubectl port-forward svc/autogpt-notification 8007:8007 -n autogpt &
NOTIFY_PID=$!

# Supabase Kong API Gateway (port 8000)
echo "🔐 Starting API gateway port forward (localhost:8000)..."
kubectl port-forward svc/supabase-kong 8000:8000 -n autogpt &
KONG_PID=$!

# Supabase Auth (port 9999)
echo "🔒 Starting auth server port forward (localhost:9999)..."
kubectl port-forward svc/supabase-auth 9999:9999 -n autogpt &
AUTH_PID=$!

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are responding
echo ""
echo "🔍 Testing service health..."

# Test Frontend
if curl -s http://localhost:3000/health >/dev/null 2>&1; then
    echo "✅ Frontend: http://localhost:3000"
else
    echo "⚠️  Frontend: Not ready yet"
fi

# Test API
if curl -s http://localhost:8006/health >/dev/null 2>&1; then
    echo "✅ API Server: http://localhost:8006"
else
    echo "⚠️  API Server: Not ready yet"
fi

# Test Executor
if curl -s http://localhost:8002/health >/dev/null 2>&1; then
    echo "✅ Executor: http://localhost:8002"
else
    echo "⚠️  Executor: Not ready yet"
fi

# Test Websocket
if curl -s http://localhost:8001/health >/dev/null 2>&1; then
    echo "✅ Websocket: ws://localhost:8001"
else
    echo "⚠️  Websocket: Not ready yet"
fi

# Test Kong Gateway
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ API Gateway: http://localhost:8000"
else
    echo "⚠️  API Gateway: Not ready yet"
fi

echo ""
echo "🎉 AutoGPT is ready!"
echo ""
echo "📋 Access URLs:"
echo "   Frontend:         http://localhost:3000"
echo "   API Server:       http://localhost:8006/api"
echo "   Executor:         http://localhost:8002"
echo "   Websocket:        ws://localhost:8001"
echo "   Scheduler:        http://localhost:8003"
echo "   Database Manager: http://localhost:8005"
echo "   Notifications:    http://localhost:8007"
echo "   API Gateway:      http://localhost:8000"
echo "   Auth Server:      http://localhost:9999"
echo ""
echo "Press Ctrl+C to stop all port forwards"

# Wait for user to stop
wait