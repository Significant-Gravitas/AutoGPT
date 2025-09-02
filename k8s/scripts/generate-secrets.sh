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
CONFIG_DIR="$K8S_ROOT/configs"

# Check if .env file exists
if [ ! -f "$CONFIG_DIR/.env" ]; then
    if [ -f "$CONFIG_DIR/.env.default" ]; then
        echo -e "${YELLOW}⚠️  .env file not found${NC}"
        echo "Copying .env.default to .env for you to customize..."
        cp "$CONFIG_DIR/.env.default" "$CONFIG_DIR/.env"
        echo ""
        echo -e "${BLUE}📝 Please edit $CONFIG_DIR/.env and update:${NC}"
        echo "  - Database passwords"
        echo "  - Your domain name"
        echo "  - API keys you want to use"
        echo "  - Security keys (generate new ones)"
        echo ""
        read -p "Press Enter after editing .env file..."
    else
        echo -e "${RED}❌ Neither .env nor .env.default found${NC}"
        exit 1
    fi
fi

# Load environment variables
echo -e "${BLUE}📂 Loading environment variables...${NC}"
export $(grep -v '^#' "$CONFIG_DIR/.env" | xargs)

# Validate required variables
REQUIRED_VARS=(
    "DB_PASS"
    "REDIS_PASSWORD" 
    "RABBITMQ_DEFAULT_PASS"
    "SUPABASE_JWT_SECRET"
    "ENCRYPTION_KEY"
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

# Generate database URL
DATABASE_URL="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}?schema=${DB_SCHEMA}&connect_timeout=${DB_CONNECT_TIMEOUT}"
DIRECT_URL="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}?schema=${DB_SCHEMA}&connect_timeout=${DB_CONNECT_TIMEOUT}"

# Create Kubernetes secrets
echo -e "${BLUE}🔑 Creating AutoGPT secrets...${NC}"
kubectl create secret generic autogpt-secrets \
    --namespace=autogpt \
    --from-literal=database-url="$DATABASE_URL" \
    --from-literal=direct-url="$DIRECT_URL" \
    --from-literal=redis-password="$REDIS_PASSWORD" \
    --from-literal=rabbitmq-password="$RABBITMQ_DEFAULT_PASS" \
    --from-literal=supabase-jwt-secret="$SUPABASE_JWT_SECRET" \
    --from-literal=encryption-key="$ENCRYPTION_KEY" \
    --from-literal=unsubscribe-secret="$UNSUBSCRIBE_SECRET_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

echo -e "${BLUE}🔑 Creating Supabase secrets...${NC}"
kubectl create secret generic supabase-secrets \
    --namespace=autogpt \
    --from-literal=SUPABASE_DB_USER="$SUPABASE_DB_USER" \
    --from-literal=SUPABASE_DB_PASS="$SUPABASE_DB_PASS" \
    --from-literal=SUPABASE_DB_NAME="$SUPABASE_DB_NAME" \
    --from-literal=SUPABASE_JWT_SECRET="$SUPABASE_JWT_SECRET" \
    --from-literal=SUPABASE_JWT_ANON_KEY="$SUPABASE_JWT_ANON_KEY" \
    --from-literal=SUPABASE_SERVICE_ROLE_KEY="$SUPABASE_SERVICE_ROLE_KEY" \
    --from-literal=password="$SUPABASE_DB_PASS" \
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