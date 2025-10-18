#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🌐 AutoGPT Local Access Setup${NC}"
echo "==============================="
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

echo -e "${BLUE}🔍 Getting LoadBalancer IPs from ingress...${NC}"

# Wait for ingresses to get IPs (they might still be provisioning)
echo "Waiting for ingress controllers to get external IPs..."
kubectl wait --for=jsonpath='{.status.loadBalancer.ingress}' ingress --all -n autogpt --timeout=300s || true

# Get ingress IPs
INGRESSES=$(kubectl get ingress -n autogpt -o jsonpath='{range .items[*]}{.metadata.name}:{.status.loadBalancer.ingress[0].ip}{"\n"}{end}')

if [ -z "$INGRESSES" ]; then
    echo -e "${YELLOW}⚠️  No external IPs found yet${NC}"
    echo "LoadBalancers may still be provisioning. Please try again in a few minutes."
    echo ""
    echo "You can check status with:"
    echo "  kubectl get ingress -n autogpt"
    exit 1
fi

echo -e "${GREEN}✅ Found LoadBalancer IPs:${NC}"
echo "$INGRESSES"
echo ""

# Generate /etc/hosts entries
echo -e "${BLUE}📝 Generating /etc/hosts entries...${NC}"

HOSTS_ENTRIES=""
# Parse domain from environment
eval "DOMAIN=$AUTOGPT_DOMAIN"

while IFS=':' read -r ingress_name ip; do
    if [ -n "$ip" ] && [ "$ip" != "null" ]; then
        case "$ingress_name" in
            *server*|*api*)
                HOSTS_ENTRIES="$HOSTS_ENTRIES$ip\tapi.autogpt.$DOMAIN\n"
                ;;
            *builder*|*frontend*)
                HOSTS_ENTRIES="$HOSTS_ENTRIES$ip\tautogpt.$DOMAIN\n"
                ;;
            *websocket*|*ws*)
                HOSTS_ENTRIES="$HOSTS_ENTRIES$ip\tws.autogpt.$DOMAIN\n"
                ;;
            *auth*|*supabase*)
                HOSTS_ENTRIES="$HOSTS_ENTRIES$ip\tauth.$DOMAIN\n"
                ;;
        esac
    fi
done <<< "$INGRESSES"

if [ -z "$HOSTS_ENTRIES" ]; then
    echo -e "${YELLOW}⚠️  No valid IPs found${NC}"
    echo "Please check that ingresses have external IPs assigned"
    exit 1
fi

echo -e "${YELLOW}📋 Add these entries to your /etc/hosts file:${NC}"
echo ""
echo -e "$HOSTS_ENTRIES"
echo ""

# Offer to automatically update /etc/hosts (requires sudo)
read -p "Would you like me to automatically add these to /etc/hosts? (requires sudo) [y/N]: " auto_update

if [[ $auto_update =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}🔧 Updating /etc/hosts (requires sudo password)...${NC}"
    
    # Create backup
    sudo cp /etc/hosts /etc/hosts.backup.$(date +%Y%m%d_%H%M%S)
    
    # Remove any existing AutoGPT entries
    sudo sed -i.tmp '/# AutoGPT Kubernetes/,/# End AutoGPT Kubernetes/d' /etc/hosts
    
    # Add new entries
    echo "" | sudo tee -a /etc/hosts > /dev/null
    echo "# AutoGPT Kubernetes" | sudo tee -a /etc/hosts > /dev/null
    echo -e "$HOSTS_ENTRIES" | sudo tee -a /etc/hosts > /dev/null
    echo "# End AutoGPT Kubernetes" | sudo tee -a /etc/hosts > /dev/null
    
    echo -e "${GREEN}✅ /etc/hosts updated successfully!${NC}"
else
    echo -e "${YELLOW}💡 Manual setup required${NC}"
    echo "Copy the entries above and add them to your /etc/hosts file"
fi

echo ""
echo -e "${GREEN}🎉 Setup complete!${NC}"
echo ""
echo -e "${BLUE}🌐 Access your AutoGPT deployment:${NC}"
echo "  Frontend:  https://autogpt.$DOMAIN"
echo "  API:       https://api.autogpt.$DOMAIN"
echo "  WebSocket: wss://ws.autogpt.$DOMAIN"
echo "  Auth:      https://auth.$DOMAIN"
echo ""
echo -e "${YELLOW}⏳ Note: SSL certificates may take a few minutes to provision${NC}"