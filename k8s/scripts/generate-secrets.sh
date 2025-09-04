#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔐 AutoGPT Secrets Generator${NC}"
echo "============================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AUTOGPT_ROOT="$(cd "$K8S_ROOT/.." && pwd)"
PLATFORM_DIR="$AUTOGPT_ROOT/autogpt_platform"

# Use .env.default from autogpt_platform directory
ENV_FILE="$PLATFORM_DIR/.env.default"

if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}❌ Environment file not found: $ENV_FILE${NC}"
    exit 1
fi

# Load environment variables
echo -e "${BLUE}📂 Loading environment variables from $ENV_FILE...${NC}"
export $(grep -v '^#' "$ENV_FILE" | xargs)

# Validate required variables
REQUIRED_VARS=(
    "POSTGRES_PASSWORD"
    "JWT_SECRET"
    "ANON_KEY"
    "SERVICE_ROLE_KEY"
)

echo -e "${BLUE}✅ Validating required variables...${NC}"
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo -e "${RED}❌ Missing required variable: $var${NC}"
        exit 1
    fi
done

# Create namespace if it doesn't exist
echo -e "${BLUE}🏗️  Creating namespace...${NC}"
kubectl create namespace autogpt --dry-run=client -o yaml | kubectl apply -f -

# Create Kubernetes secrets using .env.default variables
echo -e "${BLUE}🔑 Creating Supabase secrets...${NC}"
kubectl create secret generic supabase-secrets \
    --namespace=autogpt \
    --from-literal=SUPABASE_DB_USER="postgres" \
    --from-literal=SUPABASE_DB_PASS="$POSTGRES_PASSWORD" \
    --from-literal=SUPABASE_DB_NAME="$POSTGRES_DB" \
    --from-literal=SUPABASE_JWT_SECRET="$JWT_SECRET" \
    --from-literal=SUPABASE_JWT_ANON_KEY="$ANON_KEY" \
    --from-literal=SUPABASE_SERVICE_ROLE_KEY="$SERVICE_ROLE_KEY" \
    --from-literal=DASHBOARD_USERNAME="$DASHBOARD_USERNAME" \
    --from-literal=DASHBOARD_PASSWORD="$DASHBOARD_PASSWORD" \
    --from-literal=SMTP_USERNAME="" \
    --from-literal=SMTP_PASSWORD="" \
    --from-literal=password="$POSTGRES_PASSWORD" \
    --dry-run=client -o yaml | kubectl apply -f -

# Create optional API keys secret (if any are provided)
API_KEYS_SECRET=""
[ -n "$OPENAI_API_KEY" ] && API_KEYS_SECRET="$API_KEYS_SECRET --from-literal=openai-api-key=$OPENAI_API_KEY"
[ -n "$ANTHROPIC_API_KEY" ] && API_KEYS_SECRET="$API_KEYS_SECRET --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY"
[ -n "$GOOGLE_CLIENT_ID" ] && API_KEYS_SECRET="$API_KEYS_SECRET --from-literal=google-client-id=$GOOGLE_CLIENT_ID"
[ -n "$GOOGLE_CLIENT_SECRET" ] && API_KEYS_SECRET="$API_KEYS_SECRET --from-literal=google-client-secret=$GOOGLE_CLIENT_SECRET"

if [ -n "$API_KEYS_SECRET" ]; then
    echo -e "${BLUE}🔑 Creating API keys secret...${NC}"
    eval "kubectl create secret generic autogpt-api-keys --namespace=autogpt $API_KEYS_SECRET --dry-run=client -o yaml | kubectl apply -f -"
fi

echo -e "${GREEN}✅ Secrets created successfully!${NC}"
echo ""
echo -e "${BLUE}🔍 Secrets in namespace 'autogpt':${NC}"
kubectl get secrets -n autogpt