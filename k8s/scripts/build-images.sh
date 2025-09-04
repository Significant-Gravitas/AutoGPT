#!/bin/bash

# AutoGPT Docker Image Build Script
# This script builds AutoGPT Docker images and optionally pushes them to a registry

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables from .env if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../configs/.env"

# Save command-line arguments before sourcing .env
CLI_PROJECT_ID="${GCP_PROJECT_ID:-}"
CLI_REGION="${GCP_REGION:-}"
CLI_REGISTRY_TYPE="${REGISTRY_TYPE:-}"

if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Configuration (CLI overrides .env, .env overrides defaults)
AUTOGPT_SOURCE_DIR="${AUTOGPT_SOURCE_DIR:-}"
REGISTRY_TYPE="${CLI_REGISTRY_TYPE:-${REGISTRY_TYPE:-local}}" # Options: local, gcr, dockerhub
PROJECT_ID="${CLI_PROJECT_ID:-${GCP_PROJECT_ID:-}}"
REGION="${CLI_REGION:-${GCP_REGION:-us-central1}}"
REPOSITORY_NAME="${ARTIFACT_REPO:-autogpt}"
DOCKER_HUB_ORG="${DOCKER_HUB_ORG:-autogpt}" # Docker Hub organization/username
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_message "$BLUE" "📋 Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_message "$RED" "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if AutoGPT source directory exists
    if [ ! -d "$AUTOGPT_SOURCE_DIR" ]; then
        print_message "$RED" "❌ AutoGPT source directory not found at $AUTOGPT_SOURCE_DIR"
        print_message "$YELLOW" "Set AUTOGPT_SOURCE_DIR environment variable to the correct path"
        exit 1
    fi
    
    # Check registry-specific requirements
    case "$REGISTRY_TYPE" in
        gcr)
            if ! command -v gcloud &> /dev/null; then
                print_message "$RED" "❌ gcloud CLI is not installed. Required for GCR."
                exit 1
            fi
            if [ -z "$PROJECT_ID" ]; then
                PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
                if [ -z "$PROJECT_ID" ]; then
                    print_message "$RED" "❌ GCP Project ID not set. Please set GCP_PROJECT_ID or configure gcloud."
                    exit 1
                fi
            fi
            ;;
        dockerhub)
            print_message "$YELLOW" "⚠️  Make sure you're logged in to Docker Hub: docker login"
            ;;
        local)
            print_message "$GREEN" "✅ Building images locally (no push)"
            ;;
        *)
            print_message "$RED" "❌ Invalid REGISTRY_TYPE: $REGISTRY_TYPE"
            print_message "$YELLOW" "Valid options: local, gcr, dockerhub"
            exit 1
            ;;
    esac
    
    print_message "$GREEN" "✅ Prerequisites checked"
}

# Function to setup registry
setup_registry() {
    case "$REGISTRY_TYPE" in
        gcr)
            print_message "$BLUE" "🔧 Setting up Google Artifact Registry..."
            
            # Check if repository exists
            if gcloud artifacts repositories describe $REPOSITORY_NAME --location=$REGION &>/dev/null; then
                print_message "$GREEN" "✅ Artifact Registry repository '$REPOSITORY_NAME' already exists"
            else
                print_message "$YELLOW" "📦 Creating Artifact Registry repository..."
                gcloud artifacts repositories create $REPOSITORY_NAME \
                    --repository-format=docker \
                    --location=$REGION \
                    --description="AutoGPT Docker images"
                print_message "$GREEN" "✅ Repository created"
            fi
            
            # Configure Docker authentication
            print_message "$BLUE" "🔐 Configuring Docker authentication..."
            gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
            print_message "$GREEN" "✅ Docker authentication configured"
            ;;
        dockerhub)
            print_message "$BLUE" "🔧 Using Docker Hub registry"
            print_message "$YELLOW" "Images will be pushed to: docker.io/${DOCKER_HUB_ORG}/"
            ;;
        local)
            print_message "$BLUE" "🔧 Local build only - no registry setup needed"
            ;;
    esac
}

# Function to get image tag based on registry type
get_image_tag() {
    local service_name=$1
    
    case "$REGISTRY_TYPE" in
        gcr)
            echo "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/${service_name}:${IMAGE_TAG}"
            ;;
        dockerhub)
            echo "${DOCKER_HUB_ORG}/${service_name}:${IMAGE_TAG}"
            ;;
        local)
            echo "${service_name}:${IMAGE_TAG}"
            ;;
    esac
}

# Function to build Docker image
build_image() {
    local service_name=$1
    local dockerfile_path=$2
    local target=$3
    local context_path=$4
    
    print_message "$BLUE" "🏗️  Building $service_name image..."
    
    local image_tag=$(get_image_tag "$service_name")
    
    # Build the image
    if [ -n "$target" ]; then
        docker build -f "$dockerfile_path" \
            --platform linux/amd64 \
            --target "$target" \
            -t "$image_tag" \
            "$context_path"
    else
        docker build -f "$dockerfile_path" \
            --platform linux/amd64 \
            -t "$image_tag" \
            "$context_path"
    fi
    
    if [ $? -eq 0 ]; then
        print_message "$GREEN" "✅ Successfully built $service_name"
        echo "$image_tag"
    else
        print_message "$RED" "❌ Failed to build $service_name"
        return 1
    fi
}

# Function to push Docker image
push_image() {
    local image_tag=$1
    local service_name=$2
    
    if [ "$REGISTRY_TYPE" = "local" ]; then
        print_message "$YELLOW" "⏭️  Skipping push for local build"
        return 0
    fi
    
    print_message "$BLUE" "📤 Pushing $service_name to registry..."
    docker push "$image_tag"
    
    if [ $? -eq 0 ]; then
        print_message "$GREEN" "✅ Successfully pushed $service_name"
    else
        print_message "$RED" "❌ Failed to push $service_name"
        return 1
    fi
}

# Main execution
main() {
    print_message "$BLUE" "🚀 AutoGPT Docker Image Builder"
    echo "================================"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    print_message "$BLUE" "📝 Configuration:"
    echo "  Registry Type: $REGISTRY_TYPE"
    case "$REGISTRY_TYPE" in
        gcr)
            echo "  Project ID: $PROJECT_ID"
            echo "  Region: $REGION"
            echo "  Repository: $REPOSITORY_NAME"
            ;;
        dockerhub)
            echo "  Docker Hub Org: $DOCKER_HUB_ORG"
            ;;
    esac
    echo "  Source Dir: $AUTOGPT_SOURCE_DIR"
    echo "  Image Tag: $IMAGE_TAG"
    echo ""
    
    # Setup registry
    setup_registry
    
    # Define services to build (using arrays for compatibility)
    SERVICES_NAMES=(
        "autogpt-server"
        "autogpt-executor"
        "autogpt-websocket"
        "autogpt-scheduler"
        "autogpt-notification"
        "autogpt-database-manager"
        "autogpt-migrate"
        "autogpt-builder"
    )
    
    SERVICES_CONFIGS=(
        "backend/Dockerfile:server"
        "backend/Dockerfile:server"
        "backend/Dockerfile:server"
        "backend/Dockerfile:server"
        "backend/Dockerfile:server"
        "backend/Dockerfile:server"
        "backend/Dockerfile:migrate"
        "frontend/Dockerfile:"
    )
    
    # Build and push each service
    print_message "$BLUE" "🏗️  Building Docker images..."
    echo ""
    
    FAILED_BUILDS=""
    
    # If specific service requested, build only that one
    if [ -n "${1:-}" ] && [[ " ${SERVICES_NAMES[*]} " == *" $1 "* ]]; then
        TARGET_SERVICES=("$1")
        for i in "${!SERVICES_NAMES[@]}"; do
            if [[ "${SERVICES_NAMES[$i]}" == "$1" ]]; then
                TARGET_CONFIGS=("${SERVICES_CONFIGS[$i]}")
                break
            fi
        done
    else
        TARGET_SERVICES=("${SERVICES_NAMES[@]}")
        TARGET_CONFIGS=("${SERVICES_CONFIGS[@]}")
    fi
    
    for i in "${!TARGET_SERVICES[@]}"; do
        service="${TARGET_SERVICES[$i]}"
        IFS=':' read -r dockerfile target <<< "${TARGET_CONFIGS[$i]}"
        
        # Determine context path
        if [[ "$service" == "autogpt-builder" ]]; then
            context_path="$AUTOGPT_SOURCE_DIR"
            dockerfile_path="$AUTOGPT_SOURCE_DIR/autogpt_platform/frontend/Dockerfile"
        else
            context_path="$AUTOGPT_SOURCE_DIR"
            dockerfile_path="$AUTOGPT_SOURCE_DIR/autogpt_platform/$dockerfile"
        fi
        
        # Build the image
        image_tag=$(build_image "$service" "$dockerfile_path" "$target" "$context_path")
        
        if [ $? -eq 0 ]; then
            # Push the image
            push_image "$image_tag" "$service"
            if [ $? -ne 0 ]; then
                FAILED_BUILDS="$FAILED_BUILDS $service"
            fi
        else
            FAILED_BUILDS="$FAILED_BUILDS $service"
        fi
        
        echo ""
    done
    
    # Summary
    if [ -z "$FAILED_BUILDS" ]; then
        print_message "$GREEN" "✅ All Docker images built and pushed successfully!"
    else
        print_message "$YELLOW" "⚠️  Some builds failed: $FAILED_BUILDS"
    fi
    
    echo ""
    print_message "$BLUE" "📝 Image Tags:"
    for service in "${SERVICES_NAMES[@]}"; do
        echo "  $service: $(get_image_tag $service)"
    done
    
    echo ""
    print_message "$YELLOW" "Next steps:"
    
    case "$REGISTRY_TYPE" in
        gcr)
            echo "1. Update Helm values files with your registry path:"
            echo "   image.repository: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/SERVICE_NAME"
            ;;
        dockerhub)
            echo "1. Update Helm values files with your Docker Hub organization:"
            echo "   image.repository: ${DOCKER_HUB_ORG}/SERVICE_NAME"
            ;;
        local)
            echo "1. Images are built locally. To use in Kubernetes:"
            echo "   - Use a local registry (e.g., kind, minikube)"
            echo "   - Or push to a registry of your choice"
            ;;
    esac
    
    echo "2. Run './scripts/deploy.sh' to deploy to Kubernetes"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [SERVICE_NAME] [OPTIONS]"
        echo ""
        echo "Build and push AutoGPT Docker images to a registry"
        echo ""
        echo "Arguments:"
        echo "  SERVICE_NAME        Optional: specific service to build (autogpt-server, autogpt-builder, etc.)"
        echo ""
        echo "Environment Variables:"
        echo "  AUTOGPT_SOURCE_DIR   Path to AutoGPT source code (required)"
        echo "  REGISTRY_TYPE        Registry type: local, gcr, dockerhub (default: local)"
        echo "  GCP_PROJECT_ID       GCP Project ID (required for gcr)"
        echo "  GCP_REGION          GCP Region (default: us-central1)"
        echo "  ARTIFACT_REPO       Artifact Registry repository name (default: autogpt)"
        echo "  DOCKER_HUB_ORG      Docker Hub organization/username (default: autogpt)"
        echo "  IMAGE_TAG           Docker image tag (default: latest)"
        echo ""
        echo "Examples:"
        echo "  # Build all services locally"
        echo "  $0"
        echo ""
        echo "  # Build only autogpt-builder"
        echo "  $0 autogpt-builder"
        echo ""
        echo "  # Build and push to Google Container Registry"
        echo "  REGISTRY_TYPE=gcr GCP_PROJECT_ID=my-project $0 autogpt-builder"
        echo ""
        echo "Options:"
        echo "  -h, --help          Show this help message"
        exit 0
        ;;
    *)
        main "$1"
        ;;
esac