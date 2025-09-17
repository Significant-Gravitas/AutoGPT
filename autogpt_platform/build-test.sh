#!/bin/bash

# AutoGPT Platform Container Build Test Script
# This script tests container builds locally before CI/CD

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGISTRY="ghcr.io"
IMAGE_PREFIX="significant-gravitas/autogpt-platform"
VERSION="test"
BUILD_ARGS=""

# Functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

usage() {
    cat << EOF
AutoGPT Platform Container Build Test Script

Usage: $0 [OPTIONS] [COMPONENT]

COMPONENTS:
    backend     Build backend container only
    frontend    Build frontend container only
    all         Build both containers (default)

OPTIONS:
    -r, --registry REGISTRY     Container registry (default: ghcr.io)
    -t, --tag TAG              Image tag (default: test)
    --no-cache                 Build without cache
    --push                     Push images after build
    -h, --help                 Show this help message

EXAMPLES:
    $0                         # Build both containers
    $0 backend                 # Build backend only
    $0 --no-cache all          # Build without cache
    $0 --push frontend         # Build and push frontend

EOF
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    success "Docker is available"
}

build_backend() {
    info "Building backend container..."
    
    local image_name="$REGISTRY/$IMAGE_PREFIX-backend:$VERSION"
    local dockerfile="autogpt_platform/backend/Dockerfile"
    
    info "Building: $image_name"
    info "Dockerfile: $dockerfile"
    info "Context: ."
    info "Target: server"
    
    if docker build \
        -t "$image_name" \
        -f "$dockerfile" \
        --target server \
        $BUILD_ARGS \
        .; then
        success "Backend container built successfully: $image_name"
        
        # Test the container
        info "Testing backend container..."
        if docker run --rm -d --name autogpt-backend-test "$image_name" > /dev/null; then
            sleep 5
            if docker ps | grep -q autogpt-backend-test; then
                success "Backend container is running"
                docker stop autogpt-backend-test > /dev/null
            else
                warning "Backend container started but may have issues"
            fi
        else
            warning "Failed to start backend container for testing"
        fi
        
        return 0
    else
        error "Backend container build failed"
        return 1
    fi
}

build_frontend() {
    info "Building frontend container..."
    
    local image_name="$REGISTRY/$IMAGE_PREFIX-frontend:$VERSION"
    local dockerfile="autogpt_platform/frontend/Dockerfile"
    
    info "Building: $image_name"
    info "Dockerfile: $dockerfile"
    info "Context: ."
    info "Target: prod"
    
    if docker build \
        -t "$image_name" \
        -f "$dockerfile" \
        --target prod \
        $BUILD_ARGS \
        .; then
        success "Frontend container built successfully: $image_name"
        
        # Test the container
        info "Testing frontend container..."
        if docker run --rm -d --name autogpt-frontend-test -p 3001:3000 "$image_name" > /dev/null; then
            sleep 10
            if docker ps | grep -q autogpt-frontend-test; then
                if curl -s -o /dev/null -w "%{http_code}" http://localhost:3001 | grep -q "200\|302\|404"; then
                    success "Frontend container is responding"
                else
                    warning "Frontend container started but not responding to HTTP requests"
                fi
                docker stop autogpt-frontend-test > /dev/null
            else
                warning "Frontend container started but may have issues"
            fi
        else
            warning "Failed to start frontend container for testing"
        fi
        
        return 0
    else
        error "Frontend container build failed"
        return 1
    fi
}

push_images() {
    if [[ "$PUSH_IMAGES" == "true" ]]; then
        info "Pushing images to registry..."
        
        local backend_image="$REGISTRY/$IMAGE_PREFIX-backend:$VERSION"
        local frontend_image="$REGISTRY/$IMAGE_PREFIX-frontend:$VERSION"
        
        for image in "$backend_image" "$frontend_image"; do
            if docker images | grep -q "$image"; then
                info "Pushing $image..."
                if docker push "$image"; then
                    success "Pushed $image"
                else
                    error "Failed to push $image"
                fi
            fi
        done
    fi
}

show_images() {
    info "Built images:"
    docker images | grep "$IMAGE_PREFIX" | grep "$VERSION"
}

cleanup_test_containers() {
    # Clean up any test containers that might be left running
    docker ps -a | grep "autogpt-.*-test" | awk '{print $1}' | xargs -r docker rm -f > /dev/null 2>&1 || true
}

# Parse command line arguments
COMPONENT="all"
PUSH_IMAGES="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            VERSION="$2"
            shift 2
            ;;
        --no-cache)
            BUILD_ARGS="$BUILD_ARGS --no-cache"
            shift
            ;;
        --push)
            PUSH_IMAGES="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        backend|frontend|all)
            COMPONENT="$1"
            shift
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
info "AutoGPT Platform Container Build Test"
info "Component: $COMPONENT"
info "Registry: $REGISTRY"
info "Tag: $VERSION"

check_docker
cleanup_test_containers

# Build containers based on component selection
case "$COMPONENT" in
    backend)
        build_backend
        ;;
    frontend)
        build_frontend
        ;;
    all)
        if build_backend && build_frontend; then
            success "All containers built successfully"
        else
            error "Some container builds failed"
            exit 1
        fi
        ;;
esac

push_images
show_images
cleanup_test_containers

success "Build test completed successfully"