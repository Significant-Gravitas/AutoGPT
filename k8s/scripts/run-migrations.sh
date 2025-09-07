#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗄️  AutoGPT Database Migration${NC}"
echo "=============================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$K8S_ROOT/configs"
ENV_FILE="$CONFIG_DIR/.env"

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    echo -e "${BLUE}📂 Loading configuration from $ENV_FILE...${NC}"
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a  # disable automatic export
else
    echo -e "${RED}❌ .env file not found at $ENV_FILE${NC}"
    echo "Please create .env file from .env.example"
    exit 1
fi

# Calculate IMAGE_REPOSITORY based on REGISTRY_TYPE
if [ "$REGISTRY_TYPE" = "gcr" ]; then
    IMAGE_REPOSITORY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPO}"
    MIGRATION_IMAGE="${IMAGE_REPOSITORY}/autogpt-server:${IMAGE_TAG}"
elif [ "$REGISTRY_TYPE" = "local" ]; then
    MIGRATION_IMAGE="autogpt-server:${IMAGE_TAG}"
else
    echo -e "${RED}❌ Unsupported REGISTRY_TYPE: $REGISTRY_TYPE${NC}"
    exit 1
fi

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ kubectl not connected to cluster${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Checking if PostgreSQL is ready...${NC}"
# Find PostgreSQL pod 
POSTGRES_POD=$(kubectl get pods -n ${NAMESPACE} -l app=supabase-postgresql -o jsonpath='{.items[0].metadata.name}')
if [ -z "$POSTGRES_POD" ]; then
    echo -e "${RED}❌ PostgreSQL pod not found${NC}"
    echo "Please deploy infrastructure first: kubectl apply -f infrastructure.yaml"
    exit 1
fi

if ! kubectl wait --for=condition=ready pod "$POSTGRES_POD" -n ${NAMESPACE} --timeout=60s; then
    echo -e "${RED}❌ PostgreSQL is not ready${NC}"
    exit 1
fi

echo -e "${BLUE}🔧 Creating database schemas...${NC}"
echo "Using PostgreSQL pod: $POSTGRES_POD"

# Create database schemas if they don't exist
kubectl exec "$POSTGRES_POD" -n ${NAMESPACE} -- bash -c "PGPASSWORD=\"${POSTGRES_PASSWORD}\" psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c \"CREATE SCHEMA IF NOT EXISTS auth;\""
kubectl exec "$POSTGRES_POD" -n ${NAMESPACE} -- bash -c "PGPASSWORD=\"${POSTGRES_PASSWORD}\" psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c \"CREATE SCHEMA IF NOT EXISTS platform;\""

echo -e "${BLUE}🔧 Setting up port forwarding to PostgreSQL...${NC}"
# Kill any existing port-forward for PostgreSQL
pkill -f "port-forward.*supabase-postgresql.*5432" 2>/dev/null || true

# Start port forwarding to PostgreSQL
kubectl port-forward svc/supabase-postgresql 5432:5432 -n ${NAMESPACE} > /dev/null 2>&1 &
PORT_FORWARD_PID=$!
sleep 3

echo -e "${BLUE}🔧 Running Prisma migrations locally...${NC}"
DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}?schema=platform"

echo "Database URL: postgresql://${POSTGRES_USER}:****@localhost:5432/${POSTGRES_DB}?schema=platform"

# Check if we have the AutoGPT source directory
if [ -z "$AUTOGPT_SOURCE_DIR" ]; then
    echo -e "${RED}❌ AUTOGPT_SOURCE_DIR not set in .env${NC}"
    kill $PORT_FORWARD_PID 2>/dev/null
    exit 1
fi

if [ ! -d "$AUTOGPT_SOURCE_DIR/autogpt_platform/backend" ]; then
    echo -e "${RED}❌ AutoGPT backend not found at $AUTOGPT_SOURCE_DIR/autogpt_platform/backend${NC}"
    kill $PORT_FORWARD_PID 2>/dev/null
    exit 1
fi

# Run migrations locally (using db push to sync schema)
cd "$AUTOGPT_SOURCE_DIR/autogpt_platform/backend"
DATABASE_URL="$DATABASE_URL" DIRECT_URL="$DATABASE_URL" poetry run prisma db push --accept-data-loss

# Kill port forwarding
kill $PORT_FORWARD_PID 2>/dev/null

echo ""
echo -e "${GREEN}✅ Database migrations completed!${NC}"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo "  1. Deploy services: kubectl apply -f autogpt-services.yaml"
echo "  2. Check service status: kubectl get pods -n ${NAMESPACE}"