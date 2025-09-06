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

# Load environment variables
if [ -f "$CONFIG_DIR/.env" ]; then
    echo -e "${BLUE}📂 Loading configuration...${NC}"
    source "$CONFIG_DIR/.env"
else
    echo -e "${RED}❌ .env file not found${NC}"
    echo "Please run ./scripts/setup-gcp.sh first"
    exit 1
fi

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ kubectl not connected to cluster${NC}"
    echo "Please run ./scripts/setup-gcp.sh first to create and configure the cluster"
    exit 1
fi

echo -e "${BLUE}🔍 Checking if PostgreSQL is ready...${NC}"
if ! kubectl get pod supabase-postgresql-0 -n autogpt &> /dev/null; then
    echo -e "${RED}❌ PostgreSQL pod not found${NC}"
    echo "Please deploy the platform first: ./scripts/deploy.sh"
    exit 1
fi

if ! kubectl wait --for=condition=ready pod supabase-postgresql-0 -n autogpt --timeout=60s; then
    echo -e "${RED}❌ PostgreSQL is not ready${NC}"
    exit 1
fi

echo -e "${BLUE}🔧 Running database migrations...${NC}"

# Create database schemas if they don't exist
echo "Creating required database schemas..."
kubectl exec supabase-postgresql-0 -n autogpt -- bash -c "PGPASSWORD=\"$DB_PASS\" psql -U $DB_USER -d $DB_NAME -c 'CREATE SCHEMA IF NOT EXISTS auth;'"
kubectl exec supabase-postgresql-0 -n autogpt -- bash -c "PGPASSWORD=\"$DB_PASS\" psql -U $DB_USER -d $DB_NAME -c 'CREATE SCHEMA IF NOT EXISTS platform;'"

# Run Prisma migrations
echo "Running Prisma migrations..."
kubectl run db-migrate-job --rm -i \
    --image=us-central1-docker.pkg.dev/agpt-dev/autogpt/autogpt-server:latest \
    --env="DATABASE_URL=postgresql://${DB_USER}:${DB_PASS}@supabase-postgresql.autogpt.svc.cluster.local:${DB_PORT}/${DB_NAME}?schema=${DB_SCHEMA}&connect_timeout=${DB_CONNECT_TIMEOUT}" \
    --env="DIRECT_URL=postgresql://${DB_USER}:${DB_PASS}@supabase-postgresql.autogpt.svc.cluster.local:${DB_PORT}/${DB_NAME}?schema=${DB_SCHEMA}&connect_timeout=${DB_CONNECT_TIMEOUT}" \
    -n autogpt \
    -- bash -c "cd /app/autogpt_platform/backend && poetry run prisma migrate deploy"

echo ""
echo -e "${GREEN}✅ Database migrations completed!${NC}"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo "  1. Restart the main server: kubectl rollout restart deployment autogpt-server -n autogpt"
echo "  2. Check service status: kubectl get pods -n autogpt"