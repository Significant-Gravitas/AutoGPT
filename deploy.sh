#!/bin/bash
# Script para deployment del MVP de NEUS

set -e  # Exit on error

echo "🚀 Iniciando deployment de NEUS MVP..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Por favor, copia .env.example a .env y configura las variables"
    exit 1
fi

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker no está corriendo"
    exit 1
fi

# Build and start services
echo "📦 Building images..."
docker-compose build

echo "🎬 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check services health
echo "🏥 Checking services health..."
docker-compose ps

echo ""
echo "✅ Deployment completado!"
echo ""
echo "📍 Servicios disponibles:"
echo "   Frontend: http://localhost"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Para ver logs:"
echo "   docker-compose logs -f"
echo ""
echo "Para detener:"
echo "   docker-compose down"
