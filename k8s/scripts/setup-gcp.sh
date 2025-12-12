#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AutoGPT GCP Setup Script${NC}"
echo "==========================="
echo ""
echo "This script will:"
echo "1. Create GCP infrastructure with Terraform"
echo "2. Configure kubectl for your new cluster"
echo "3. Deploy AutoGPT services with Helm"
echo ""

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TERRAFORM_DIR="$K8S_ROOT/terraform/gcp"
CONFIG_DIR="$K8S_ROOT/configs"

# Load environment variables from .env if it exists
ENV_FILE="$CONFIG_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo -e "${GREEN}✅ Loaded configuration from .env${NC}"
fi

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"
for cmd in terraform gcloud kubectl helm; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}❌ $cmd is not installed${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✅ All tools found${NC}"

# Check GCP authentication
echo -e "${BLUE}🔐 Checking GCP authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
    echo -e "${RED}❌ Please authenticate with GCP first:${NC}"
    echo "  gcloud auth login"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}⚠️  No default project set${NC}"
    echo "Available projects:"
    gcloud projects list --format="table(projectId,name)"
    echo ""
    read -p "Enter your GCP project ID: " PROJECT_ID
    gcloud config set project "$PROJECT_ID"
fi

echo -e "${GREEN}✅ Using GCP project: $PROJECT_ID${NC}"
echo ""

# Check/create .env file
if [ ! -f "$CONFIG_DIR/.env" ]; then
    echo -e "${YELLOW}⚠️  Configuration file not found${NC}"
    echo "Creating .env from defaults..."
    cp "$CONFIG_DIR/.env.default" "$CONFIG_DIR/.env"
    
    # Update project ID in .env
    sed -i.bak "s/your-gcp-project-id/$PROJECT_ID/g" "$CONFIG_DIR/.env"
    rm "$CONFIG_DIR/.env.bak"
    
    echo ""
    echo -e "${RED}🚨 IMPORTANT: Edit $CONFIG_DIR/.env before continuing${NC}"
    echo "Update:"
    echo "  - AUTOGPT_DOMAIN (your actual domain name)"
    echo "  - DB_PASS (secure database password)"
    echo "  - All other passwords and API keys"
    echo ""
    read -p "Press Enter after editing .env..."
fi

# Load environment variables
source "$CONFIG_DIR/.env"

# Setup Terraform
echo -e "${BLUE}🏗️  Setting up infrastructure...${NC}"
cd "$TERRAFORM_DIR"

# Check/create terraform.tfvars
if [ ! -f "terraform.tfvars" ]; then
    echo "Creating terraform.tfvars from template..."
    cp terraform.tfvars.example terraform.tfvars
    
    # Update values from .env
    sed -i.bak "s/your-gcp-project-id/$PROJECT_ID/g" terraform.tfvars
    sed -i.bak "s/us-central1/${GCP_REGION:-us-central1}/g" terraform.tfvars
    sed -i.bak "s/CHANGE-THIS-TO-A-SECURE-PASSWORD/$DB_PASS/g" terraform.tfvars
    sed -i.bak "s/\${var\.project_id}/$PROJECT_ID/g" terraform.tfvars
    rm terraform.tfvars.bak
    
    echo -e "${YELLOW}📝 Review terraform.tfvars if needed${NC}"
fi

# Initialize and apply Terraform
echo -e "${BLUE}🔧 Initializing Terraform...${NC}"
terraform init

echo -e "${BLUE}📋 Planning infrastructure...${NC}"
terraform plan

echo ""
echo -e "${YELLOW}⚠️  This will create GCP resources that may incur charges${NC}"
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo -e "${BLUE}🚀 Creating infrastructure...${NC}"
terraform apply -auto-approve

# Configure kubectl
echo -e "${BLUE}🔑 Configuring kubectl...${NC}"
CLUSTER_NAME=$(terraform output -raw cluster_name)
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="us-central1-a"

# Deploy AutoGPT
echo -e "${BLUE}🤖 Deploying AutoGPT...${NC}"
cd "$K8S_ROOT"
./scripts/deploy.sh

echo ""
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}📝 Next Steps:${NC}"
echo "1. Update your DNS to point to the static IPs:"
terraform output static_ips
echo ""
echo "2. Wait for SSL certificates to be provisioned (5-10 minutes)"
echo "3. Access AutoGPT at: https://autogpt.$AUTOGPT_DOMAIN"