# AutoGPT Platform on Unraid

This guide covers deploying AutoGPT Platform on Unraid using the Community Applications plugin.

## Prerequisites

1. **Unraid 6.8+** (recommended 6.10+)
2. **Community Applications** plugin installed
3. **Minimum 4GB RAM** allocated to Docker
4. **10GB+ free disk space**

## Installation Methods

### Method 1: Community Applications (Recommended)

1. Open **Apps** tab in Unraid
2. Search for "AutoGPT Platform"
3. Click **Install** on the official template
4. Configure the parameters (see below)
5. Click **Apply**

### Method 2: Manual Docker Template

If the template isn't available yet, you can create it manually:

1. Go to **Docker** tab
2. Click **Add Container**
3. Use the configuration below

## Container Configuration

### AutoGPT Platform Backend

```yaml
Repository: ghcr.io/significant-gravitas/autogpt-platform-backend:latest
Network Type: Custom: autogpt
WebUI: http://[IP]:[PORT:8000]
Icon URL: https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/master/assets/autogpt_logo.png
```

#### Port Mappings
- **Container Port**: 8000
- **Host Port**: 8000 (or your preferred port)
- **Connection Type**: TCP

#### Volume Mappings
- **Container Path**: /app/data
- **Host Path**: /mnt/user/appdata/autogpt-platform/backend
- **Access Mode**: Read/Write

#### Environment Variables
```bash
DATABASE_URL=postgresql://autogpt:password@autogpt-db:5432/autogpt
REDIS_HOST=autogpt-redis
RABBITMQ_HOST=autogpt-rabbitmq
SUPABASE_URL=http://autogpt-auth:8000
```

### AutoGPT Platform Frontend

```yaml
Repository: ghcr.io/significant-gravitas/autogpt-platform-frontend:latest
Network Type: Custom: autogpt
WebUI: http://[IP]:[PORT:3000]
```

#### Port Mappings
- **Container Port**: 3000
- **Host Port**: 3000 (or your preferred port)
- **Connection Type**: TCP

#### Environment Variables
```bash
AGPT_SERVER_URL=http://[UNRAID_IP]:8000/api
SUPABASE_URL=http://[UNRAID_IP]:8001
```

## Required Dependencies

You'll also need these containers for a complete setup:

### PostgreSQL Database
```yaml
Repository: postgres:15
Network Type: Custom: autogpt
Environment Variables:
  POSTGRES_DB=autogpt
  POSTGRES_USER=autogpt
  POSTGRES_PASSWORD=your_secure_password
Volume Mappings:
  /var/lib/postgresql/data -> /mnt/user/appdata/autogpt-platform/postgres
```

### Redis
```yaml
Repository: redis:7-alpine
Network Type: Custom: autogpt
Command: redis-server --requirepass your_redis_password
Volume Mappings:
  /data -> /mnt/user/appdata/autogpt-platform/redis
```

### RabbitMQ
```yaml
Repository: rabbitmq:3-management
Network Type: Custom: autogpt
Port Mappings:
  5672:5672 (AMQP)
  15672:15672 (Management UI)
Environment Variables:
  RABBITMQ_DEFAULT_USER=autogpt
  RABBITMQ_DEFAULT_PASS=your_rabbitmq_password
Volume Mappings:
  /var/lib/rabbitmq -> /mnt/user/appdata/autogpt-platform/rabbitmq
```

## Network Setup

1. **Create Custom Network**:
   ```bash
   docker network create autogpt
   ```

2. **Assign all containers** to the `autogpt` network

## Docker Compose Alternative

For easier management, you can use docker-compose:

1. **Enable docker-compose plugin** in Unraid
2. **Create compose file** in `/mnt/user/appdata/autogpt-platform/docker-compose.yml`:

```yaml
version: '3.8'
networks:
  autogpt:
    driver: bridge

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: autogpt
      POSTGRES_USER: autogpt
      POSTGRES_PASSWORD: your_secure_password
    volumes:
      - /mnt/user/appdata/autogpt-platform/postgres:/var/lib/postgresql/data
    networks:
      - autogpt

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass your_redis_password
    volumes:
      - /mnt/user/appdata/autogpt-platform/redis:/data
    networks:
      - autogpt

  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: autogpt
      RABBITMQ_DEFAULT_PASS: your_rabbitmq_password
    volumes:
      - /mnt/user/appdata/autogpt-platform/rabbitmq:/var/lib/rabbitmq
    ports:
      - "15672:15672"
    networks:
      - autogpt

  backend:
    image: ghcr.io/significant-gravitas/autogpt-platform-backend:latest
    environment:
      DATABASE_URL: postgresql://autogpt:your_secure_password@postgres:5432/autogpt
      REDIS_HOST: redis
      RABBITMQ_HOST: rabbitmq
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - rabbitmq
    networks:
      - autogpt

  frontend:
    image: ghcr.io/significant-gravitas/autogpt-platform-frontend:latest
    environment:
      AGPT_SERVER_URL: http://[UNRAID_IP]:8000/api
    ports:
      - "3000:3000"
    depends_on:
      - backend
    networks:
      - autogpt
```

## Backup Strategy

### Important Data Locations
- **Database**: `/mnt/user/appdata/autogpt-platform/postgres`
- **User uploads**: `/mnt/user/appdata/autogpt-platform/backend`
- **Configuration**: Container templates

### Automated Backup Script
```bash
#!/bin/bash
# Save as /mnt/user/scripts/backup-autogpt.sh

BACKUP_DIR="/mnt/user/backups/autogpt-$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Stop containers
docker stop autogpt-backend autogpt-frontend

# Backup database
docker exec autogpt-postgres pg_dump -U autogpt autogpt > "$BACKUP_DIR/database.sql"

# Backup appdata
cp -r /mnt/user/appdata/autogpt-platform "$BACKUP_DIR/"

# Start containers
docker start autogpt-postgres autogpt-redis autogpt-rabbitmq autogpt-backend autogpt-frontend

echo "Backup completed: $BACKUP_DIR"
```

## Troubleshooting

### Common Issues

1. **Containers won't start**
   - Check Docker log: **Docker** tab → container → **Logs**
   - Verify network connectivity between containers
   - Ensure sufficient RAM allocated to Docker

2. **Database connection errors**
   - Verify PostgreSQL container is running
   - Check DATABASE_URL environment variable
   - Ensure containers are on same network

3. **Frontend can't reach backend**
   - Verify AGPT_SERVER_URL uses correct Unraid IP
   - Check firewall settings
   - Ensure backend container is accessible

### Performance Optimization

1. **Use SSD for appdata** if possible
2. **Allocate sufficient RAM** to Docker (minimum 4GB)
3. **Enable Docker logging limits**:
   ```json
   {
     "log-driver": "json-file",
     "log-opts": {
       "max-size": "10m",
       "max-file": "3"
     }
   }
   ```

## Monitoring

### Built-in Monitoring
- **Docker tab**: Container status and resource usage
- **RabbitMQ Management**: http://[UNRAID_IP]:15672

### Optional: Grafana Dashboard
Install Grafana from Community Applications for advanced monitoring:
1. Install **Grafana** and **Prometheus**
2. Configure scraping of container metrics
3. Import AutoGPT Platform dashboard

## Security Recommendations

1. **Change default passwords** for all services
2. **Use strong passwords** (20+ characters)
3. **Limit network access** to trusted devices only
4. **Regular updates**: Check for container updates monthly
5. **Backup encryption**: Encrypt backup files

## Updates

### Manual Updates
1. **Stop containers**: Docker tab → Stop
2. **Force update**: Docker tab → Force Update
3. **Start containers**: Docker tab → Start

### Automated Updates (Optional)
Install **Watchtower** from Community Applications for automatic updates:
```yaml
# Add to docker-compose.yml
watchtower:
  image: containrrr/watchtower
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
  command: --interval 30 --cleanup
```

## Support

- **Unraid Forums**: [AutoGPT Platform Support](https://forums.unraid.net)
- **GitHub Issues**: [AutoGPT Repository](https://github.com/Significant-Gravitas/AutoGPT/issues)
- **Discord**: [AutoGPT Community](https://discord.gg/autogpt)