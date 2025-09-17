# AutoGPT Platform Container Deployment

This guide covers deploying AutoGPT Platform using pre-built containers from GitHub Container Registry (GHCR) or Docker Hub.

## Available Container Images

The AutoGPT Platform is published as separate containers for each component:

### GitHub Container Registry (Recommended)
- **Backend**: `ghcr.io/significant-gravitas/autogpt-platform-backend:latest`
- **Frontend**: `ghcr.io/significant-gravitas/autogpt-platform-frontend:latest`

### Docker Hub
- **Backend**: `significantgravitas/autogpt-platform-backend:latest`
- **Frontend**: `significantgravitas/autogpt-platform-frontend:latest`

## Quick Start with Docker Compose

The simplest way to deploy the platform is using the provided docker-compose file:

```bash
# Download the compose file
curl -O https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/master/autogpt_platform/docker-compose.yml

# Start the platform with published containers
AUTOGPT_USE_PUBLISHED_IMAGES=true docker compose up -d
```

## Manual Container Deployment

### Prerequisites

1. **PostgreSQL Database** with pgvector extension
2. **Redis** for caching and session management
3. **RabbitMQ** for task queuing
4. **ClamAV** for file scanning (optional but recommended)

### Environment Variables

Both containers require configuration through environment variables. See the [environment configuration guide](./advanced_setup.md#environment-variables) for detailed settings.

#### Backend Container
```bash
docker run -d \
  --name autogpt-backend \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@db:5432/autogpt" \
  -e REDIS_HOST="redis" \
  -e RABBITMQ_HOST="rabbitmq" \
  ghcr.io/significant-gravitas/autogpt-platform-backend:latest
```

#### Frontend Container
```bash
docker run -d \
  --name autogpt-frontend \
  -p 3000:3000 \
  -e AGPT_SERVER_URL="http://backend:8000/api" \
  -e SUPABASE_URL="http://auth:8000" \
  ghcr.io/significant-gravitas/autogpt-platform-frontend:latest
```

## Image Versions and Tags

- `latest` - Latest stable release
- `v1.0.0` - Specific version tags
- `master` - Latest development build (use with caution)

## Health Checks

The containers include health check endpoints:

- **Backend**: `GET /health` on port 8000
- **Frontend**: HTTP 200 response on port 3000

## Resource Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB

### Recommended for Production
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 50GB+ (depends on usage)

## Security Considerations

1. **Never expose internal services** (database, Redis, RabbitMQ) to the internet
2. **Use environment files** for sensitive configuration
3. **Enable TLS** for production deployments
4. **Regular updates** - monitor for security updates
5. **Network segmentation** - isolate platform from other services

## Troubleshooting

### Common Issues

1. **Container won't start**: Check logs with `docker logs <container_name>`
2. **Database connection fails**: Verify DATABASE_URL and network connectivity
3. **Frontend can't reach backend**: Check AGPT_SERVER_URL configuration
4. **Performance issues**: Monitor resource usage and scale accordingly

### Logging

Containers log to stdout/stderr by default. Configure log aggregation for production:

```bash
# View logs
docker logs autogpt-backend
docker logs autogpt-frontend

# Follow logs
docker logs -f autogpt-backend
```

## Production Deployment Checklist

- [ ] Database backup strategy in place
- [ ] Monitoring and alerting configured
- [ ] TLS certificates configured
- [ ] Environment variables secured
- [ ] Resource limits set
- [ ] Log aggregation configured
- [ ] Health checks enabled
- [ ] Update strategy defined

## Next Steps

- [Unraid Deployment Guide](./deployment/unraid.md)
- [Home Assistant Add-on](./deployment/home-assistant.md)
- [Kubernetes Deployment](./deployment/kubernetes.md)