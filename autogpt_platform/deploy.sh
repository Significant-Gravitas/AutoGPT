#!/bin/bash

# AutoGPT Platform Deployment Script
# This script deploys AutoGPT Platform using published container images

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.published.yml"
ENV_FILE=".env"
BACKUP_DIR="backups"
LOG_FILE="deploy.log"

# Default values
REGISTRY="ghcr.io"
IMAGE_PREFIX="significant-gravitas/autogpt-platform"
VERSION="latest"
PROFILE="local"
ACTION=""

# Functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

usage() {
    cat << EOF
AutoGPT Platform Deployment Script

Usage: $0 [OPTIONS] ACTION

ACTIONS:
    deploy      Deploy the platform
    start       Start existing deployment
    stop        Stop the deployment
    restart     Restart the deployment
    update      Update to latest images
    backup      Create backup of data
    restore     Restore from backup
    logs        Show logs
    status      Show deployment status
    cleanup     Remove all containers and volumes

OPTIONS:
    -r, --registry REGISTRY     Container registry (default: ghcr.io)
    -v, --version VERSION       Image version/tag (default: latest)
    -p, --profile PROFILE       Docker compose profile (default: local)
    -f, --file FILE            Compose file (default: docker-compose.published.yml)
    -e, --env FILE             Environment file (default: .env)
    -h, --help                 Show this help message

EXAMPLES:
    $0 deploy                                    # Deploy with defaults
    $0 -v v1.0.0 deploy                         # Deploy specific version
    $0 -r docker.io update                      # Update from Docker Hub
    $0 -p production deploy                     # Deploy for production

EOF
}

check_dependencies() {
    info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    success "All dependencies are available"
}

setup_environment() {
    info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p "$BACKUP_DIR"
    mkdir -p "data/postgres"
    mkdir -p "data/redis"
    mkdir -p "data/rabbitmq"
    mkdir -p "data/backend"
    
    # Create environment file if it doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        info "Creating default environment file..."
        cat > "$ENV_FILE" << EOF
# AutoGPT Platform Configuration
POSTGRES_PASSWORD=your-super-secret-and-long-postgres-password
REDIS_PASSWORD=your-redis-password
RABBITMQ_PASSWORD=your-rabbitmq-password
JWT_SECRET=your-long-random-jwt-secret-with-at-least-32-characters

# Registry Configuration
REGISTRY=${REGISTRY}
IMAGE_PREFIX=${IMAGE_PREFIX}
VERSION=${VERSION}

# Network Configuration
BACKEND_PORT=8006
FRONTEND_PORT=3000
POSTGRES_PORT=5432
REDIS_PORT=6379
RABBITMQ_PORT=5672
RABBITMQ_MANAGEMENT_PORT=15672

# Development
PROFILE=${PROFILE}
EOF
        warning "Created default $ENV_FILE - please review and update passwords!"
    fi
    
    success "Environment setup complete"
}

check_ports() {
    info "Checking if required ports are available..."
    
    local ports=(3000 8000 8001 8002 8003 8005 8006 8007 5432 6379 5672 15672)
    local used_ports=()
    
    for port in "${ports[@]}"; do
        if ss -tuln | grep -q ":$port "; then
            used_ports+=("$port")
        fi
    done
    
    if [[ ${#used_ports[@]} -gt 0 ]]; then
        warning "The following ports are already in use: ${used_ports[*]}"
        warning "This may cause conflicts. Please stop services using these ports or modify the configuration."
    else
        success "All required ports are available"
    fi
}

pull_images() {
    info "Pulling container images..."
    
    local images=(
        "$REGISTRY/$IMAGE_PREFIX-backend:$VERSION"
        "$REGISTRY/$IMAGE_PREFIX-frontend:$VERSION"
    )
    
    for image in "${images[@]}"; do
        info "Pulling $image..."
        if docker pull "$image"; then
            success "Pulled $image"
        else
            error "Failed to pull $image"
            exit 1
        fi
    done
}

deploy() {
    info "Deploying AutoGPT Platform..."
    
    check_dependencies
    setup_environment
    check_ports
    pull_images
    
    # Update compose file with current settings
    export REGISTRY="$REGISTRY"
    export IMAGE_PREFIX="$IMAGE_PREFIX"
    export VERSION="$VERSION"
    
    info "Starting services..."
    if docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d; then
        success "AutoGPT Platform deployed successfully!"
        
        info "Waiting for services to be ready..."
        sleep 10
        
        show_status
        
        info "Access the platform at:"
        info "  Frontend: http://localhost:3000"
        info "  Backend API: http://localhost:8006"
        info "  Database Admin: http://localhost:8910 (if using local profile)"
        info "  RabbitMQ Management: http://localhost:15672"
    else
        error "Deployment failed. Check logs with: $0 logs"
        exit 1
    fi
}

start_services() {
    info "Starting AutoGPT Platform services..."
    if docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" start; then
        success "Services started successfully"
        show_status
    else
        error "Failed to start services"
        exit 1
    fi
}

stop_services() {
    info "Stopping AutoGPT Platform services..."
    if docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" stop; then
        success "Services stopped successfully"
    else
        error "Failed to stop services"
        exit 1
    fi
}

restart_services() {
    info "Restarting AutoGPT Platform services..."
    stop_services
    start_services
}

update_services() {
    info "Updating AutoGPT Platform to version $VERSION..."
    
    # Pull new images
    pull_images
    
    # Recreate containers with new images
    info "Recreating containers with new images..."
    if docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d --force-recreate; then
        success "Update completed successfully"
        show_status
    else
        error "Update failed"
        exit 1
    fi
}

backup_data() {
    local backup_name="autogpt-backup-$(date +%Y%m%d-%H%M%S)"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    info "Creating backup: $backup_name..."
    mkdir -p "$backup_path"
    
    # Stop services for consistent backup
    info "Stopping services for backup..."
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" stop
    
    # Backup database
    info "Backing up database..."
    docker compose -f "$COMPOSE_FILE" run --rm db pg_dump -U postgres postgres > "$backup_path/database.sql"
    
    # Backup volumes
    info "Backing up data volumes..."
    cp -r data "$backup_path/"
    
    # Backup configuration
    cp "$ENV_FILE" "$backup_path/"
    cp "$COMPOSE_FILE" "$backup_path/"
    
    # Restart services
    info "Restarting services..."
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" start
    
    success "Backup created: $backup_path"
}

restore_data() {
    if [[ $# -lt 1 ]]; then
        error "Please specify backup directory to restore from"
        error "Usage: $0 restore <backup-directory>"
        exit 1
    fi
    
    local backup_path="$1"
    
    if [[ ! -d "$backup_path" ]]; then
        error "Backup directory not found: $backup_path"
        exit 1
    fi
    
    warning "This will overwrite current data. Are you sure? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        info "Restore cancelled"
        exit 0
    fi
    
    info "Restoring from backup: $backup_path..."
    
    # Stop services
    stop_services
    
    # Restore data
    info "Restoring data volumes..."
    rm -rf data
    cp -r "$backup_path/data" .
    
    # Restore configuration
    if [[ -f "$backup_path/$ENV_FILE" ]]; then
        cp "$backup_path/$ENV_FILE" .
        info "Restored environment configuration"
    fi
    
    # Start services
    start_services
    
    # Restore database
    if [[ -f "$backup_path/database.sql" ]]; then
        info "Restoring database..."
        docker compose -f "$COMPOSE_FILE" exec -T db psql -U postgres postgres < "$backup_path/database.sql"
    fi
    
    success "Restore completed successfully"
}

show_logs() {
    info "Showing logs (press Ctrl+C to exit)..."
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" logs -f
}

show_status() {
    info "AutoGPT Platform Status:"
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" ps
    
    echo
    info "Service Health:"
    
    # Check service health
    local services=("frontend:3000" "rest_server:8006" "db:5432" "redis:6379")
    
    for service in "${services[@]}"; do
        local name="${service%:*}"
        local port="${service#*:}"
        
        if docker compose -f "$COMPOSE_FILE" ps "$name" | grep -q "Up"; then
            if nc -z localhost "$port" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} $name (port $port)"
            else
                echo -e "  ${YELLOW}⚠${NC} $name (container up, port not accessible)"
            fi
        else
            echo -e "  ${RED}✗${NC} $name (container down)"
        fi
    done
}

cleanup() {
    warning "This will remove all containers and volumes. Are you sure? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        info "Cleanup cancelled"
        exit 0
    fi
    
    info "Cleaning up AutoGPT Platform..."
    
    # Stop and remove containers
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" down -v --remove-orphans
    
    # Remove images
    docker images | grep "$IMAGE_PREFIX" | awk '{print $3}' | xargs -r docker rmi
    
    # Remove data directories
    rm -rf data
    
    success "Cleanup completed"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -f|--file)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        -e|--env)
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        deploy|start|stop|restart|update|backup|restore|logs|status|cleanup)
            ACTION="$1"
            shift
            break
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if action is provided
if [[ -z "$ACTION" ]]; then
    error "No action specified"
    usage
    exit 1
fi

# Execute action
case "$ACTION" in
    deploy)
        deploy
        ;;
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    update)
        update_services
        ;;
    backup)
        backup_data
        ;;
    restore)
        restore_data "$@"
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    *)
        error "Unknown action: $ACTION"
        usage
        exit 1
        ;;
esac