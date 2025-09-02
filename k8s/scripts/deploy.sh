#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AutoGPT Kubernetes Deployment${NC}"
echo "================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$K8S_ROOT/configs"

# Function to check command existence
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        echo "Please install $1 and try again."
        exit 1
    fi
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"
check_command kubectl
check_command helm
echo -e "${GREEN}✅ Prerequisites found${NC}"
echo ""

# Check if .env exists
if [ ! -f "$CONFIG_DIR/.env" ]; then
    echo -e "${YELLOW}⚠️  Configuration file not found${NC}"
    echo "Creating .env from template..."
    
    if [ -f "$CONFIG_DIR/.env.default" ]; then
        cp "$CONFIG_DIR/.env.default" "$CONFIG_DIR/.env"
        echo ""
        echo -e "${RED}🚨 IMPORTANT: Edit $CONFIG_DIR/.env before continuing${NC}"
        echo "Update at minimum:"
        echo "  - AUTOGPT_DOMAIN (your actual domain)"  
        echo "  - DB_PASS (secure database password)"
        echo "  - All passwords and secret keys"
        echo ""
        read -p "Press Enter after editing .env..."
    else
        echo -e "${RED}❌ .env.default not found${NC}"
        exit 1
    fi
fi

# Load environment variables
echo -e "${BLUE}📂 Loading configuration...${NC}"
source "$CONFIG_DIR/.env"

# Validate domain is set
if [ -z "$AUTOGPT_DOMAIN" ] || [ "$AUTOGPT_DOMAIN" = "yourdomain.com" ]; then
    echo -e "${RED}❌ Please set AUTOGPT_DOMAIN in .env file${NC}"
    exit 1
fi

# Generate secrets
echo -e "${BLUE}🔐 Generating Kubernetes secrets...${NC}"
./scripts/generate-secrets.sh

# Add Helm repositories for dependencies
echo -e "${BLUE}📦 Adding Helm repositories...${NC}"
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Deploy infrastructure services first (PostgreSQL, Redis, RabbitMQ, Supabase)
echo -e "${BLUE}🗄️  Deploying infrastructure services...${NC}"

# Deploy Supabase (includes PostgreSQL)
echo -e "${BLUE}📦 Deploying Supabase (Auth + Database)...${NC}"
helm upgrade --install supabase ./helm/supabase \
    --namespace autogpt \
    --set global.domain="$AUTOGPT_DOMAIN" \
    --set auth.environment.API_EXTERNAL_URL="https://auth.$AUTOGPT_DOMAIN" \
    --set auth.environment.GOTRUE_SITE_URL="https://auth.$AUTOGPT_DOMAIN"

# Deploy Redis
echo -e "${BLUE}📦 Deploying Redis...${NC}"
helm upgrade --install redis ./helm/redis \
    --namespace autogpt \
    --set auth.password="$REDIS_PASSWORD"

# Deploy RabbitMQ  
echo -e "${BLUE}📦 Deploying RabbitMQ...${NC}"
helm upgrade --install rabbitmq ./helm/rabbit-mq \
    --namespace autogpt \
    --set auth.username="$RABBITMQ_DEFAULT_USER" \
    --set auth.password="$RABBITMQ_DEFAULT_PASS"

# Wait for infrastructure services
echo -e "${BLUE}⏳ Waiting for infrastructure services...${NC}"
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql -n autogpt --timeout=600s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n autogpt --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=rabbitmq -n autogpt --timeout=300s

# Deploy AutoGPT services
echo -e "${BLUE}🤖 Deploying AutoGPT services...${NC}"

# Deploy in order of dependencies
services=("autogpt-database-manager" "autogpt-server" "autogpt-scheduler" "autogpt-notification" "autogpt-websocket" "autogpt-builder")

for service in "${services[@]}"; do
    echo -e "${BLUE}📦 Deploying $service...${NC}"
    helm upgrade --install "$service" "./helm/$service" \
        --namespace autogpt \
        --set global.domain="$AUTOGPT_DOMAIN" \
        --set image.repository="${IMAGE_REGISTRY:-autogpt}/$service" \
        --set domain="$AUTOGPT_DOMAIN"
done

# Wait for all deployments
echo -e "${BLUE}⏳ Waiting for all services to be ready...${NC}"
kubectl wait --for=condition=available --timeout=600s deployment --all -n autogpt

# Display deployment status
echo ""
echo -e "${GREEN}🎉 Deployment completed!${NC}"
echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
kubectl get pods -n autogpt
echo ""
echo -e "${BLUE}🌐 Services:${NC}"
kubectl get services -n autogpt
echo ""
echo -e "${BLUE}🔗 Ingress:${NC}"
kubectl get ingress -n autogpt

echo ""
echo -e "${GREEN}🎉 AutoGPT Platform is now running!${NC}"
echo ""
echo -e "${BLUE}📱 Access your deployment:${NC}"
echo "  Frontend: https://autogpt.$AUTOGPT_DOMAIN"
echo "  API:      https://api.autogpt.$AUTOGPT_DOMAIN"
echo "  WebSocket: wss://ws.autogpt.$AUTOGPT_DOMAIN"
echo ""
echo -e "${YELLOW}⚠️  Don't forget to:${NC}"
echo "  1. Point your DNS records to the static IPs"
echo "  2. Wait for SSL certificates to be provisioned"
echo "  3. Check that all pods are running: kubectl get pods -n autogpt"