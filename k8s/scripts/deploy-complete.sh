#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AutoGPT Complete Deployment${NC}"
echo "==============================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_DIR="$K8S_ROOT/configs"
ENV_FILE="$CONFIG_DIR/.env"

# Change to k8s root directory
cd "$K8S_ROOT"

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
check_command envsubst
echo -e "${GREEN}✅ Prerequisites found${NC}"

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    echo -e "${BLUE}📂 Loading configuration from $ENV_FILE...${NC}"
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a  # disable automatic export
else
    echo -e "${RED}❌ .env file not found at $ENV_FILE${NC}"
    echo "Please create .env file from .env.example:"
    echo "cp $CONFIG_DIR/.env.example $CONFIG_DIR/.env"
    exit 1
fi

# Export all environment variables for envsubst
export NAMESPACE POSTGRES_HOST POSTGRES_DB POSTGRES_USER POSTGRES_PORT POSTGRES_PASSWORD
export DB_HOST DB_USER DB_PASS DB_NAME DB_PORT DB_CONNECTION_LIMIT DB_CONNECT_TIMEOUT DB_POOL_TIMEOUT DB_SCHEMA
export REDIS_HOST REDIS_PASSWORD REDIS_PORT
export RABBITMQ_HOST RABBITMQ_PORT RABBITMQ_DEFAULT_USER RABBITMQ_DEFAULT_PASS
export JWT_SECRET ANON_KEY SERVICE_ROLE_KEY JWT_VERIFY_KEY GOTRUE_JWT_SECRET
export ENCRYPTION_KEY UNSUBSCRIBE_SECRET_KEY
export AGENTSERVER_HOST SCHEDULER_HOST DATABASEMANAGER_HOST EXECUTIONMANAGER_HOST NOTIFICATIONMANAGER_HOST CLAMAV_SERVICE_HOST PYRO_HOST
export SUPABASE_URL SUPABASE_SERVICE_ROLE_KEY PLATFORM_BASE_URL FRONTEND_BASE_URL CORS_ALLOWED_ORIGINS
export NEXT_PUBLIC_SUPABASE_URL NEXT_PUBLIC_SUPABASE_ANON_KEY NEXT_PUBLIC_AGPT_SERVER_URL NEXT_PUBLIC_AGPT_WS_SERVER_URL
export NEXT_PUBLIC_FRONTEND_BASE_URL NEXT_PUBLIC_APP_ENV NEXT_PUBLIC_BEHAVE_AS NEXT_PUBLIC_LAUNCHDARKLY_ENABLED
export NEXT_PUBLIC_LAUNCHDARKLY_CLIENT_ID NEXT_PUBLIC_SHOW_BILLING_PAGE NEXT_PUBLIC_TURNSTILE NEXT_PUBLIC_REACT_QUERY_DEVTOOL NEXT_PUBLIC_GA_MEASUREMENT_ID
export REGISTRY_TYPE GCP_PROJECT_ID GCP_REGION ARTIFACT_REPO IMAGE_TAG AUTOGPT_SOURCE_DIR AUTOGPT_DOMAIN
export MEDIA_GCS_BUCKET_NAME

# Calculate IMAGE_REPOSITORY based on REGISTRY_TYPE
if [ "$REGISTRY_TYPE" = "gcr" ]; then
    export IMAGE_REPOSITORY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/${ARTIFACT_REPO}"
    export FULL_SERVER_IMAGE="${IMAGE_REPOSITORY}/autogpt-server:${IMAGE_TAG}"
    export FULL_BUILDER_IMAGE="${IMAGE_REPOSITORY}/autogpt-builder:${IMAGE_TAG}"
elif [ "$REGISTRY_TYPE" = "local" ]; then
    export IMAGE_REPOSITORY=""
    export FULL_SERVER_IMAGE="autogpt-server:${IMAGE_TAG}"
    export FULL_BUILDER_IMAGE="autogpt-builder:${IMAGE_TAG}"
else
    echo -e "${RED}❌ Unsupported REGISTRY_TYPE: $REGISTRY_TYPE${NC}"
    echo "Supported values: gcr, local"
    exit 1
fi

echo -e "${BLUE}📝 Using images:${NC}"
echo "  Server: $FULL_SERVER_IMAGE"
echo "  Builder: $FULL_BUILDER_IMAGE"

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ kubectl not connected to cluster${NC}"
    echo "Please configure kubectl to connect to your cluster first."
    exit 1
fi

# Build images if requested
if [ "${1:-}" = "--build" ]; then
    echo -e "${BLUE}🏗️  Building AutoGPT images...${NC}"
    export AUTOGPT_SOURCE_DIR REGISTRY_TYPE GCP_PROJECT_ID GCP_REGION IMAGE_TAG
    ./scripts/build-images.sh
fi

# Create namespace
echo -e "${BLUE}🏗️  Creating namespace...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Deploy infrastructure with environment variables
echo -e "${BLUE}🗄️  Deploying infrastructure services...${NC}"
envsubst < infrastructure.yaml | kubectl apply -f -

echo -e "${BLUE}⏳ Waiting for infrastructure services to be ready...${NC}"
echo "Waiting for PostgreSQL..."
kubectl wait --for=condition=ready pod -l app=supabase-postgresql -n ${NAMESPACE} --timeout=120s

echo "Waiting for Redis..."
kubectl wait --for=condition=ready pod -l app=redis,role=master -n ${NAMESPACE} --timeout=60s

echo "Waiting for RabbitMQ..."
kubectl wait --for=condition=ready pod -l app=rabbitmq -n ${NAMESPACE} --timeout=60s

echo -e "${GREEN}✅ Infrastructure services ready${NC}"

# Run database migrations
echo -e "${BLUE}🔧 Running database migrations...${NC}"
./scripts/run-migrations.sh

# Deploy AutoGPT services with environment variables
echo -e "${BLUE}🚀 Deploying AutoGPT services...${NC}"
envsubst < autogpt-services.yaml | kubectl apply -f -

echo -e "${BLUE}⏳ Waiting for AutoGPT services to start...${NC}"
sleep 15

# Show deployment status
echo -e "${BLUE}📊 Deployment Status:${NC}"
kubectl get pods -n ${NAMESPACE}
echo ""
kubectl get services -n ${NAMESPACE}

echo ""
echo -e "${GREEN}✅ Deployment completed!${NC}"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo "  1. Check pod status: kubectl get pods -n ${NAMESPACE}"
echo "  2. View logs: kubectl logs -f deployment/autogpt-server -n ${NAMESPACE}"
echo "  3. Set up port forwarding:"
echo "     kubectl port-forward svc/autogpt-builder 3000:3000 -n ${NAMESPACE} &"
echo "     kubectl port-forward svc/autogpt-server 8006:8006 -n ${NAMESPACE} &"
echo "  4. Access the application at http://localhost:3000"