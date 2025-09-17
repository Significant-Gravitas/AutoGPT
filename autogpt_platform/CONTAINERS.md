# AutoGPT Platform Container Publishing

This document describes the container publishing infrastructure and deployment options for the AutoGPT Platform.

## Published Container Images

### GitHub Container Registry (GHCR) - Recommended

- **Backend**: `ghcr.io/significant-gravitas/autogpt-platform-backend`
- **Frontend**: `ghcr.io/significant-gravitas/autogpt-platform-frontend`

### Docker Hub

- **Backend**: `significantgravitas/autogpt-platform-backend`
- **Frontend**: `significantgravitas/autogpt-platform-frontend`

## Available Tags

- `latest` - Latest stable release from master branch
- `v1.0.0`, `v1.1.0`, etc. - Specific version releases
- `main` - Latest development build (use with caution)

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository (or just download the compose file)
git clone https://github.com/Significant-Gravitas/AutoGPT.git
cd AutoGPT/autogpt_platform

# Deploy with published images
./deploy.sh deploy
```

### Manual Docker Run

```bash
# Start dependencies first
docker network create autogpt

# PostgreSQL
docker run -d --name postgres --network autogpt \
  -e POSTGRES_DB=autogpt \
  -e POSTGRES_USER=autogpt \
  -e POSTGRES_PASSWORD=password \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15

# Redis
docker run -d --name redis --network autogpt \
  -v redis_data:/data \
  redis:7-alpine redis-server --requirepass password

# RabbitMQ
docker run -d --name rabbitmq --network autogpt \
  -e RABBITMQ_DEFAULT_USER=autogpt \
  -e RABBITMQ_DEFAULT_PASS=password \
  -p 15672:15672 \
  rabbitmq:3-management

# Backend
docker run -d --name backend --network autogpt \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://autogpt:password@postgres:5432/autogpt \
  -e REDIS_HOST=redis \
  -e RABBITMQ_HOST=rabbitmq \
  ghcr.io/significant-gravitas/autogpt-platform-backend:latest

# Frontend
docker run -d --name frontend --network autogpt \
  -p 3000:3000 \
  -e AGPT_SERVER_URL=http://localhost:8000/api \
  ghcr.io/significant-gravitas/autogpt-platform-frontend:latest
```

## Deployment Scripts

### Deploy Script

The included `deploy.sh` script provides a complete deployment solution:

```bash
# Basic deployment
./deploy.sh deploy

# Deploy specific version
./deploy.sh -v v1.0.0 deploy

# Deploy from Docker Hub
./deploy.sh -r docker.io deploy

# Production deployment
./deploy.sh -p production deploy

# Other operations
./deploy.sh start     # Start services
./deploy.sh stop      # Stop services
./deploy.sh restart   # Restart services
./deploy.sh update    # Update to latest
./deploy.sh backup    # Create backup
./deploy.sh status    # Show status
./deploy.sh logs      # Show logs
./deploy.sh cleanup   # Remove everything
```

## Platform-Specific Deployment Guides

### Unraid

See [Unraid Deployment Guide](../docs/content/platform/deployment/unraid.md)

Key features:
- Community Applications template
- Web UI management
- Automatic updates
- Built-in backup system

### Home Assistant Add-on

See [Home Assistant Add-on Guide](../docs/content/platform/deployment/home-assistant.md)

Key features:
- Native Home Assistant integration
- Automation services
- Entity monitoring
- Backup integration

### Kubernetes

See [Kubernetes Deployment Guide](../docs/content/platform/deployment/kubernetes.md)

Key features:
- Helm charts
- Horizontal scaling
- Health checks
- Persistent volumes

## Container Architecture

### Backend Container

- **Base Image**: `debian:13-slim`
- **Runtime**: Python 3.13 with Poetry
- **Services**: REST API, WebSocket, Executor, Scheduler, Database Manager, Notification
- **Ports**: 8000-8007 (depending on service)
- **Health Check**: `GET /health`

### Frontend Container

- **Base Image**: `node:21-alpine`
- **Runtime**: Next.js production build
- **Port**: 3000
- **Health Check**: HTTP 200 on root path

## Environment Configuration

### Required Environment Variables

#### Backend
```env
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_HOST=redis
RABBITMQ_HOST=rabbitmq
JWT_SECRET=your-secret-key
```

#### Frontend
```env
AGPT_SERVER_URL=http://backend:8000/api
SUPABASE_URL=http://auth:8000
```

### Optional Configuration

```env
# Logging
LOG_LEVEL=INFO
ENABLE_DEBUG=false

# Performance
REDIS_PASSWORD=your-redis-password
RABBITMQ_PASSWORD=your-rabbitmq-password

# Security
CORS_ORIGINS=http://localhost:3000
```

## CI/CD Pipeline

### GitHub Actions Workflow

The publishing workflow (`.github/workflows/platform-container-publish.yml`) automatically:

1. **Triggers** on releases and manual dispatch
2. **Builds** both backend and frontend containers
3. **Tests** container functionality
4. **Publishes** to both GHCR and Docker Hub
5. **Tags** with version and latest

### Manual Publishing

```bash
# Build and tag locally
docker build -t ghcr.io/significant-gravitas/autogpt-platform-backend:latest \
  -f autogpt_platform/backend/Dockerfile \
  --target server .

docker build -t ghcr.io/significant-gravitas/autogpt-platform-frontend:latest \
  -f autogpt_platform/frontend/Dockerfile \
  --target prod .

# Push to registry
docker push ghcr.io/significant-gravitas/autogpt-platform-backend:latest
docker push ghcr.io/significant-gravitas/autogpt-platform-frontend:latest
```

## Security Considerations

### Container Security

1. **Non-root users** - Containers run as non-root
2. **Minimal base images** - Using slim/alpine images
3. **No secrets in images** - All secrets via environment variables
4. **Read-only filesystem** - Where possible
5. **Resource limits** - CPU and memory limits set

### Deployment Security

1. **Network isolation** - Use dedicated networks
2. **TLS encryption** - Enable HTTPS in production
3. **Secret management** - Use Docker secrets or external secret stores
4. **Regular updates** - Keep images updated
5. **Vulnerability scanning** - Regular security scans

## Monitoring

### Health Checks

All containers include health checks:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' container_name

# Manual health check
curl http://localhost:8000/health
```

### Metrics

The backend exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

### Logging

Containers log to stdout/stderr for easy aggregation:

```bash
# View logs
docker logs container_name

# Follow logs
docker logs -f container_name

# Aggregate logs
docker compose logs -f
```

## Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   # Check logs
   docker logs container_name
   
   # Check environment
   docker exec container_name env
   ```

2. **Database connection failed**
   ```bash
   # Test connectivity
   docker exec backend ping postgres
   
   # Check database status
   docker exec postgres pg_isready
   ```

3. **Port conflicts**
   ```bash
   # Check port usage
   ss -tuln | grep :3000
   
   # Use different ports
   docker run -p 3001:3000 ...
   ```

### Debug Mode

Enable debug mode for detailed logging:

```env
LOG_LEVEL=DEBUG
ENABLE_DEBUG=true
```

## Performance Optimization

### Resource Limits

```yaml
# Docker Compose
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Scaling

```bash
# Scale backend services
docker compose up -d --scale backend=3

# Or use Docker Swarm
docker service scale backend=3
```

## Backup and Recovery

### Data Backup

```bash
# Database backup
docker exec postgres pg_dump -U autogpt autogpt > backup.sql

# Volume backup
docker run --rm -v postgres_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres_backup.tar.gz /data
```

### Restore

```bash
# Database restore
docker exec -i postgres psql -U autogpt autogpt < backup.sql

# Volume restore
docker run --rm -v postgres_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/postgres_backup.tar.gz -C /
```

## Support

- **Documentation**: [Platform Docs](../docs/content/platform/)
- **Issues**: [GitHub Issues](https://github.com/Significant-Gravitas/AutoGPT/issues)
- **Discord**: [AutoGPT Community](https://discord.gg/autogpt)
- **Docker Hub**: [Container Registry](https://hub.docker.com/r/significantgravitas/)

## Contributing

To contribute to the container infrastructure:

1. **Test locally** with `docker build` and `docker run`
2. **Update documentation** if making changes
3. **Test deployment scripts** on your platform
4. **Submit PR** with clear description of changes

## Roadmap

- [ ] ARM64 support for Apple Silicon
- [ ] Helm charts for Kubernetes
- [ ] Official Unraid template
- [ ] Home Assistant Add-on store submission
- [ ] Multi-stage builds optimization
- [ ] Security scanning integration
- [ ] Performance benchmarking