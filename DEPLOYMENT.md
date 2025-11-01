# Guía de Deployment a Producción - NEUS MVP

Esta guía te ayudará a desplegar el MVP de NEUS en un entorno de producción usando un VPS (Virtual Private Server).

## Tabla de Contenidos

1. [Requisitos Previos](#requisitos-previos)
2. [Configuración del VPS](#configuración-del-vps)
3. [Instalación de Dependencias](#instalación-de-dependencias)
4. [Configuración de Dominio y DNS](#configuración-de-dominio-y-dns)
5. [Deployment con Docker](#deployment-con-docker)
6. [Configuración de HTTPS con Let's Encrypt](#configuración-de-https-con-lets-encrypt)
7. [Nginx como Reverse Proxy](#nginx-como-reverse-proxy)
8. [Firewall y Seguridad](#firewall-y-seguridad)
9. [Backup de Base de Datos](#backup-de-base-de-datos)
10. [Monitoreo y Logs](#monitoreo-y-logs)
11. [Actualización de la Aplicación](#actualización-de-la-aplicación)
12. [Troubleshooting](#troubleshooting)

## Requisitos Previos

### Recursos Mínimos Recomendados

- **CPU**: 2 vCPUs
- **RAM**: 4 GB
- **Disco**: 40 GB SSD
- **Sistema Operativo**: Ubuntu 22.04 LTS

### Servicios Necesarios

- VPS (DigitalOcean, AWS EC2, Linode, etc.)
- Dominio registrado (ej: neus.com)
- Anthropic API Key

### Proveedores Recomendados

- **DigitalOcean**: Droplet de $24/mes (2 vCPUs, 4GB RAM)
- **AWS EC2**: t3.medium ($30/mes)
- **Linode**: Linode 4GB ($24/mes)
- **Hetzner**: CX21 (€5.83/mes) - Muy económico

## Configuración del VPS

### 1. Crear VPS

**DigitalOcean (ejemplo)**:
```bash
# Desde la interfaz web de DigitalOcean:
1. Create Droplet
2. Seleccionar: Ubuntu 22.04 LTS
3. Plan: Basic, Regular, 4GB RAM, 2 vCPUs
4. Datacenter: Closest to your target users
5. Authentication: SSH Key (recomendado) o Password
6. Create Droplet
```

### 2. Conectarse al VPS

```bash
# Reemplaza YOUR_SERVER_IP con la IP de tu VPS
ssh root@YOUR_SERVER_IP
```

### 3. Configuración Inicial de Seguridad

```bash
# Actualizar sistema
apt update && apt upgrade -y

# Crear usuario no-root
adduser neus
usermod -aG sudo neus

# Configurar SSH para el nuevo usuario
rsync --archive --chown=neus:neus ~/.ssh /home/neus

# Deshabilitar login de root (opcional, recomendado)
nano /etc/ssh/sshd_config
# Cambiar: PermitRootLogin no
systemctl restart sshd

# Ahora usa el nuevo usuario
exit
ssh neus@YOUR_SERVER_IP
```

### 4. Configurar Firewall UFW

```bash
# Permitir SSH
sudo ufw allow OpenSSH

# Permitir HTTP y HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Habilitar firewall
sudo ufw enable

# Verificar estado
sudo ufw status
```

## Instalación de Dependencias

### 1. Instalar Docker

```bash
# Actualizar paquetes
sudo apt update

# Instalar dependencias previas
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# Agregar repositorio de Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Instalar Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Verificar instalación
docker --version

# Agregar usuario al grupo docker
sudo usermod -aG docker ${USER}

# Aplicar cambios (o logout/login)
newgrp docker

# Verificar que funciona sin sudo
docker run hello-world
```

### 2. Instalar Docker Compose

```bash
# Descargar Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Dar permisos de ejecución
sudo chmod +x /usr/local/bin/docker-compose

# Verificar instalación
docker-compose --version
```

### 3. Instalar Git

```bash
sudo apt install -y git
git --version
```

### 4. Instalar Nginx (para reverse proxy)

```bash
sudo apt install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx
sudo systemctl status nginx
```

## Configuración de Dominio y DNS

### 1. Apuntar Dominio al VPS

En tu proveedor de dominio (GoDaddy, Namecheap, Cloudflare, etc.):

```
# Agregar registros DNS:
Tipo    Nombre    Valor              TTL
A       @         YOUR_SERVER_IP     300
A       www       YOUR_SERVER_IP     300
```

### 2. Verificar Propagación DNS

```bash
# Desde tu computadora local
nslookup neus.com
dig neus.com

# Debe mostrar tu IP del VPS
```

Puede tomar hasta 48 horas, pero usualmente es inmediato.

## Deployment con Docker

### 1. Clonar Repositorio

```bash
# Crear directorio para la aplicación
cd /home/neus
mkdir apps
cd apps

# Opción A: Si tienes Git repo
git clone https://github.com/your-org/neus.git
cd neus

# Opción B: Transferir archivos manualmente
# Desde tu máquina local:
# rsync -avz -e ssh /path/to/neus/ neus@YOUR_SERVER_IP:/home/neus/apps/neus/
```

### 2. Configurar Variables de Entorno

```bash
cd /home/neus/apps/neus

# Copiar template
cp .env.example .env

# Editar con tus credenciales de producción
nano .env
```

Configuración `.env` para producción:

```env
# Database
DB_PASSWORD=VERY_SECURE_PASSWORD_HERE_123!@#

# Backend API Keys
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_REAL_KEY_HERE

# Production (opcional)
PRODUCTION_URL=https://neus.com
```

**IMPORTANTE**: Usa contraseñas seguras y NUNCA compartas el archivo `.env`.

### 3. Modificar docker-compose.yml para Producción

```bash
nano docker-compose.yml
```

Actualizar la sección del frontend para usar el dominio real:

```yaml
  frontend:
    build:
      context: ./frontend
      args:
        VITE_API_URL: https://api.neus.com  # O https://neus.com/api
    container_name: neus-frontend
    depends_on:
      - backend
    ports:
      - "3000:80"  # Cambiar a puerto interno
    networks:
      - neus-network
    restart: unless-stopped

  backend:
    build: ./backend
    container_name: neus-backend
    depends_on:
      db:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://neus:${DB_PASSWORD}@db:5432/neus
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      CORS_ORIGINS: https://neus.com,https://www.neus.com
    ports:
      - "8000:8000"
    networks:
      - neus-network
    restart: unless-stopped
```

### 4. Ejecutar Deployment

```bash
# Construir imágenes
docker-compose build

# Iniciar servicios
docker-compose up -d

# Verificar estado
docker-compose ps

# Ver logs
docker-compose logs -f
```

## Configuración de HTTPS con Let's Encrypt

### 1. Instalar Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 2. Detener Nginx temporalmente

```bash
sudo systemctl stop nginx
```

### 3. Obtener Certificados SSL

```bash
# Reemplaza neus.com con tu dominio
sudo certbot certonly --standalone -d neus.com -d www.neus.com

# Seguir las instrucciones:
# - Ingresar email
# - Aceptar términos
# - Decidir si compartir email con EFF (opcional)

# Los certificados se guardan en:
# /etc/letsencrypt/live/neus.com/fullchain.pem
# /etc/letsencrypt/live/neus.com/privkey.pem
```

### 4. Configurar Renovación Automática

```bash
# Probar renovación
sudo certbot renew --dry-run

# Certbot automáticamente crea un cronjob para renovación
# Verificar:
sudo systemctl status certbot.timer
```

Los certificados se renovarán automáticamente antes de expirar (cada 90 días).

## Nginx como Reverse Proxy

### 1. Configurar Nginx

```bash
# Crear archivo de configuración
sudo nano /etc/nginx/sites-available/neus
```

Contenido del archivo:

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name neus.com www.neus.com;

    return 301 https://$server_name$request_uri;
}

# HTTPS Server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name neus.com www.neus.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/neus.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/neus.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Frontend (React App)
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Documentación API
    location /docs {
        proxy_pass http://localhost:8000/docs;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Client max body size (para uploads)
    client_max_body_size 10M;

    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/json;
}
```

### 2. Activar Configuración

```bash
# Crear symlink
sudo ln -s /etc/nginx/sites-available/neus /etc/nginx/sites-enabled/

# Eliminar configuración default (opcional)
sudo rm /etc/nginx/sites-enabled/default

# Probar configuración
sudo nginx -t

# Recargar Nginx
sudo systemctl restart nginx

# Verificar estado
sudo systemctl status nginx
```

### 3. Verificar Deployment

Visita https://neus.com en tu navegador. Deberías ver:
- Certificado SSL válido (candado verde)
- Frontend cargando correctamente
- Chatbot funcionando
- Formularios funcionando

## Firewall y Seguridad

### 1. Configuración UFW (ya hecho anteriormente)

```bash
sudo ufw status verbose
```

Debería mostrar:
```
Status: active

To                         Action      From
--                         ------      ----
OpenSSH                    ALLOW       Anywhere
80/tcp                     ALLOW       Anywhere
443/tcp                    ALLOW       Anywhere
```

### 2. Fail2Ban (Protección contra brute force)

```bash
# Instalar Fail2Ban
sudo apt install -y fail2ban

# Crear configuración personalizada
sudo nano /etc/fail2ban/jail.local
```

Contenido:
```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
```

```bash
# Reiniciar Fail2Ban
sudo systemctl restart fail2ban

# Verificar estado
sudo fail2ban-client status
```

### 3. Actualizar Regularmente

```bash
# Crear script de actualización
nano ~/update-system.sh
```

Contenido:
```bash
#!/bin/bash
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt autoclean
echo "System updated on $(date)" >> ~/update.log
```

```bash
# Hacer ejecutable
chmod +x ~/update-system.sh

# Agregar a cron (ejecutar cada domingo a las 3 AM)
crontab -e

# Agregar línea:
0 3 * * 0 /home/neus/update-system.sh
```

## Backup de Base de Datos

### 1. Script de Backup Manual

```bash
# Crear directorio de backups
mkdir -p /home/neus/backups

# Crear script de backup
nano /home/neus/backup-db.sh
```

Contenido:
```bash
#!/bin/bash

# Configuración
BACKUP_DIR="/home/neus/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="neus_backup_${DATE}.sql"
KEEP_DAYS=7

# Crear backup
docker-compose -f /home/neus/apps/neus/docker-compose.yml exec -T db pg_dump -U neus neus > "${BACKUP_DIR}/${BACKUP_FILE}"

# Comprimir
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Eliminar backups antiguos (más de 7 días)
find ${BACKUP_DIR} -name "neus_backup_*.sql.gz" -mtime +${KEEP_DAYS} -delete

echo "Backup completado: ${BACKUP_FILE}.gz"
echo "Backup realizado el $(date)" >> ${BACKUP_DIR}/backup.log
```

```bash
# Hacer ejecutable
chmod +x /home/neus/backup-db.sh

# Ejecutar manualmente
./backup-db.sh
```

### 2. Automatizar Backups con Cron

```bash
# Editar crontab
crontab -e

# Agregar línea para backup diario a las 2 AM
0 2 * * * /home/neus/backup-db.sh
```

### 3. Backup Remoto (opcional pero recomendado)

**Opción A: Usar rclone para subir a S3/Google Drive/Dropbox**

```bash
# Instalar rclone
curl https://rclone.org/install.sh | sudo bash

# Configurar (seguir wizard)
rclone config

# Modificar script de backup para incluir upload
# Agregar al final de backup-db.sh:
rclone copy ${BACKUP_DIR}/${BACKUP_FILE}.gz remote:neus-backups/
```

### 4. Restaurar Backup

```bash
# Detener aplicación
cd /home/neus/apps/neus
docker-compose down

# Restaurar desde backup
gunzip -c /home/neus/backups/neus_backup_YYYYMMDD_HHMMSS.sql.gz | docker-compose exec -T db psql -U neus -d neus

# Reiniciar aplicación
docker-compose up -d
```

## Monitoreo y Logs

### 1. Logs de Docker

```bash
# Ver logs en tiempo real
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f db

# Ver últimas 100 líneas
docker-compose logs --tail=100 backend
```

### 2. Logs de Nginx

```bash
# Access logs
sudo tail -f /var/log/nginx/access.log

# Error logs
sudo tail -f /var/log/nginx/error.log

# Logs filtrados (errores 5xx)
sudo grep "HTTP/[12].[01]\" [5]" /var/log/nginx/access.log
```

### 3. Logs de Sistema

```bash
# Ver logs del sistema
sudo journalctl -f

# Logs de Docker
sudo journalctl -u docker.service -f

# Logs de Nginx
sudo journalctl -u nginx.service -f
```

### 4. Monitoreo de Recursos

```bash
# Ver uso de CPU, memoria, disco
htop

# Si no está instalado
sudo apt install -y htop

# Ver uso de disco
df -h

# Ver contenedores y recursos
docker stats

# Ver espacio de Docker
docker system df
```

### 5. Herramientas de Monitoreo (Avanzado)

**Opción 1: Portainer (GUI para Docker)**

```bash
docker volume create portainer_data
docker run -d -p 9000:9000 --name=portainer --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce

# Acceder: https://neus.com:9000
```

**Opción 2: Grafana + Prometheus (Métricas avanzadas)**

Configuración compleja, ver: https://grafana.com/docs/

**Opción 3: Uptime Kuma (Monitoring simple)**

```bash
docker run -d --restart=always -p 3001:3001 -v uptime-kuma:/app/data --name uptime-kuma louislam/uptime-kuma:1

# Acceder: http://YOUR_SERVER_IP:3001
```

## Actualización de la Aplicación

### 1. Actualización Básica (Git Pull)

```bash
cd /home/neus/apps/neus

# Pull últimos cambios
git pull origin main

# Reconstruir imágenes
docker-compose build

# Reiniciar servicios con zero-downtime (aproximado)
docker-compose up -d --force-recreate
```

### 2. Actualización con Rollback

```bash
cd /home/neus/apps/neus

# Backup actual
docker-compose exec db pg_dump -U neus neus > ~/backups/pre-update_$(date +%Y%m%d).sql

# Tag actual de imágenes
docker tag neus-backend:latest neus-backend:backup
docker tag neus-frontend:latest neus-frontend:backup

# Pull cambios
git pull origin main

# Reconstruir
docker-compose build

# Reiniciar
docker-compose up -d

# Si hay problemas, hacer rollback:
# docker-compose down
# docker tag neus-backend:backup neus-backend:latest
# docker tag neus-frontend:backup neus-frontend:latest
# docker-compose up -d
```

### 3. Blue-Green Deployment (Avanzado)

Requiere configuración adicional con múltiples instancias y load balancer.

## Troubleshooting

### Problema: Sitio no accesible

**Diagnóstico**:
```bash
# Verificar que Nginx está corriendo
sudo systemctl status nginx

# Verificar que contenedores están corriendo
docker-compose ps

# Verificar logs de Nginx
sudo tail -f /var/log/nginx/error.log

# Verificar firewall
sudo ufw status
```

**Soluciones**:
```bash
# Reiniciar Nginx
sudo systemctl restart nginx

# Reiniciar contenedores
docker-compose restart

# Verificar configuración de Nginx
sudo nginx -t
```

### Problema: Error 502 Bad Gateway

**Causa**: Backend no está respondiendo

**Diagnóstico**:
```bash
# Verificar backend
docker-compose logs backend

# Verificar que backend está corriendo
docker-compose ps backend

# Verificar conectividad
curl http://localhost:8000/api/health
```

**Soluciones**:
```bash
# Reiniciar backend
docker-compose restart backend

# Reconstruir backend
docker-compose build backend
docker-compose up -d backend
```

### Problema: Base de datos no conecta

**Diagnóstico**:
```bash
# Verificar PostgreSQL
docker-compose logs db
docker-compose ps db

# Intentar conectar manualmente
docker-compose exec db psql -U neus -d neus
```

**Soluciones**:
```bash
# Reiniciar DB
docker-compose restart db

# Verificar variables de entorno
docker-compose exec backend env | grep DATABASE_URL
```

### Problema: Certificado SSL expirado

**Diagnóstico**:
```bash
# Verificar fecha de expiración
sudo certbot certificates
```

**Solución**:
```bash
# Renovar manualmente
sudo certbot renew

# Reiniciar Nginx
sudo systemctl restart nginx
```

### Problema: Disco lleno

**Diagnóstico**:
```bash
# Ver uso de disco
df -h

# Ver uso de Docker
docker system df

# Ver logs grandes
du -sh /var/log/*
```

**Solución**:
```bash
# Limpiar Docker
docker system prune -a

# Limpiar logs viejos
sudo journalctl --vacuum-time=7d

# Limpiar backups viejos
find /home/neus/backups -name "*.sql.gz" -mtime +30 -delete
```

## Checklist de Deployment

### Pre-Deployment
- [ ] VPS creado y configurado
- [ ] Dominio configurado y DNS propagado
- [ ] Dependencias instaladas (Docker, Docker Compose, Nginx)
- [ ] Firewall configurado (UFW)
- [ ] Variables de entorno configuradas (.env)
- [ ] Certificados SSL obtenidos

### Deployment
- [ ] Código desplegado (Git clone o rsync)
- [ ] Docker Compose build exitoso
- [ ] Contenedores iniciados correctamente
- [ ] Nginx configurado y corriendo
- [ ] HTTPS funcionando correctamente

### Post-Deployment
- [ ] Sitio accesible desde navegador
- [ ] Formularios funcionando
- [ ] Chatbot respondiendo
- [ ] API docs accesibles
- [ ] Backups automáticos configurados
- [ ] Monitoreo configurado
- [ ] Fail2Ban activado
- [ ] Actualizaciones automáticas configuradas

### Testing
- [ ] Frontend carga correctamente
- [ ] Backend API responde
- [ ] Base de datos acepta conexiones
- [ ] Formulario de contacto funciona
- [ ] Formulario de diagnóstico funciona
- [ ] Chatbot genera respuestas
- [ ] HTTPS sin errores
- [ ] Redirección HTTP -> HTTPS funciona

## Costos Estimados

| Servicio | Costo Mensual | Anual |
|----------|---------------|-------|
| VPS (DigitalOcean 4GB) | $24 | $288 |
| Dominio (.com) | ~$1 | ~$12 |
| Anthropic API (estimado) | $10-50 | $120-600 |
| **TOTAL** | **$35-75** | **$420-900** |

**Notas**:
- Let's Encrypt SSL es GRATIS
- Nginx es GRATIS
- Costos de API varían según uso
- Puedes reducir costos con VPS más económicos (Hetzner, Linode)

## Recursos Adicionales

- **DigitalOcean Tutorials**: https://www.digitalocean.com/community/tutorials
- **Docker Docs**: https://docs.docker.com/
- **Nginx Docs**: https://nginx.org/en/docs/
- **Let's Encrypt**: https://letsencrypt.org/
- **Certbot**: https://certbot.eff.org/
- **FastAPI Deployment**: https://fastapi.tiangolo.com/deployment/

---

**Desarrollado por el equipo de NEUS** | 2025
