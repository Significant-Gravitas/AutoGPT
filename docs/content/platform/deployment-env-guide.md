# AutoGPT Platform - Environment Variables Deployment Guide

This guide documents all environment variables that **MUST** be updated when deploying AutoGPT Platform to a different PC/server.

## Overview

The AutoGPT Platform consists of three main components that require environment configuration:
- **Platform Configuration** (Supabase & shared settings) - `.env.default` → `.env`
- **Backend Services** - `backend/.env.default` → `backend/.env`
- **Frontend Application** - `frontend/.env.default` → `frontend/.env`

## Critical Variables to Update for Deployment

### 1. URL and Host Configuration

These are the **most critical** variables that must be changed from `localhost` to your actual domain/IP:

#### Frontend (`frontend/.env`)
```bash
# Supabase connection (must match your Supabase deployment)
NEXT_PUBLIC_SUPABASE_URL=http://localhost:8000  # → https://your-domain.com:8000

# Backend API endpoints
NEXT_PUBLIC_AGPT_SERVER_URL=http://localhost:8006/api  # → https://your-domain.com:8006/api
NEXT_PUBLIC_AGPT_WS_SERVER_URL=ws://localhost:8001/ws  # → wss://your-domain.com:8001/ws

# Frontend base URL (for callbacks and redirects)
NEXT_PUBLIC_FRONTEND_BASE_URL=http://localhost:3000  # → https://your-domain.com
```

#### Backend (`backend/.env`)
```bash
# Database connection
DB_HOST=localhost  # → your-database-host.com

# Redis cache
REDIS_HOST=localhost  # → your-redis-host.com

# Supabase authentication
SUPABASE_URL=http://localhost:8000  # → https://your-supabase-instance.com

# Platform URLs (critical for OAuth callbacks and webhooks)
PLATFORM_BASE_URL=http://localhost:8000  # → https://your-domain.com:8000
FRONTEND_BASE_URL=http://localhost:3000  # → https://your-domain.com
```

#### Platform/Supabase (`.env`)
```bash
# Authentication & redirects
SITE_URL=http://localhost:3000  # → https://your-domain.com
API_EXTERNAL_URL=http://localhost:8000  # → https://your-domain.com:8000
SUPABASE_PUBLIC_URL=http://localhost:8000  # → https://your-domain.com:8000
```

### 2. Security Keys (MUST be changed for production)

These default keys are **insecure** and must be regenerated for production:

#### Platform/Supabase (`.env`)
```bash
# Generate new secure passwords/keys
POSTGRES_PASSWORD=your-super-secret-and-long-postgres-password  # → Generate new
JWT_SECRET=your-super-secret-jwt-token-with-at-least-32-characters-long  # → Generate new
ANON_KEY=<default-key>  # → Generate with Supabase CLI
SERVICE_ROLE_KEY=<default-key>  # → Generate with Supabase CLI
DASHBOARD_PASSWORD=this_password_is_insecure_and_should_be_updated  # → Generate new
SECRET_KEY_BASE=<default-key>  # → Generate new
VAULT_ENC_KEY=your-encryption-key-32-chars-min  # → Generate new
```

#### Backend (`backend/.env`)
```bash
# Must match the Platform/Supabase configuration
DB_PASS=your-super-secret-and-long-postgres-password  # → Same as POSTGRES_PASSWORD
SUPABASE_SERVICE_ROLE_KEY=<service-role-key>  # → Same as SERVICE_ROLE_KEY
JWT_VERIFY_KEY=<jwt-secret>  # → Same as JWT_SECRET

# Generate new keys for:
ENCRYPTION_KEY=<generate-new>  # Use: from cryptography.fernet import Fernet;Fernet.generate_key().decode()
UNSUBSCRIBE_SECRET_KEY=<generate-new>  # Generate similarly

# RabbitMQ credentials
RABBITMQ_DEFAULT_PASS=<generate-new>  # → Generate secure password
```

#### Frontend (`frontend/.env`)
```bash
# Must match Platform configuration
NEXT_PUBLIC_SUPABASE_ANON_KEY=<anon-key>  # → Same as ANON_KEY from Platform
```

### 3. OAuth Callback URLs

All OAuth integrations must have their callback URLs updated to match your deployment:

#### Common OAuth Callback Pattern
Replace `http://localhost:3000` with your frontend URL:
```
http://localhost:3000/auth/integrations/oauth_callback
→ https://your-domain.com/auth/integrations/oauth_callback
```

This applies to all OAuth providers in `backend/.env`:
- GitHub OAuth
- Google OAuth
- Twitter/X OAuth
- Discord OAuth
- Linear OAuth
- Todoist OAuth
- Notion OAuth
- Reddit OAuth

### 4. Database Configuration

#### Backend (`backend/.env`)
```bash
# Database connection pooling (adjust based on server capacity)
DB_CONNECTION_LIMIT=12  # Adjust based on expected load
DB_CONNECT_TIMEOUT=60  # Increase if remote database
DB_POOL_TIMEOUT=300  # Adjust for network latency

# Redis configuration
# REDIS_PASSWORD=  # Optional - only set if your Redis instance requires authentication
```

### 5. Port Configuration

If using non-standard ports, update these:

#### Platform/Supabase (`.env`)
```bash
KONG_HTTP_PORT=8000  # API Gateway port
KONG_HTTPS_PORT=8443  # API Gateway SSL port
STUDIO_PORT=3000  # Supabase Studio port
```

#### Backend (`backend/.env`)
```bash
DB_PORT=5432  # PostgreSQL port
REDIS_PORT=6379  # Redis port
```

### 6. Environment-Specific Settings

#### Frontend (`frontend/.env`)
```bash
# Change for production
NEXT_PUBLIC_APP_ENV=local  # → production
NEXT_PUBLIC_BEHAVE_AS=LOCAL  # → CLOUD or SELF_HOSTED

# Disable development tools in production
NEXT_PUBLIC_REACT_QUERY_DEVTOOL=true  # → false

# Enable for production monitoring
NEXT_PUBLIC_LAUNCHDARKLY_ENABLED=false  # → true (if using feature flags)
NEXT_PUBLIC_SHOW_BILLING_PAGE=false  # → true (if using billing)
NEXT_PUBLIC_TURNSTILE=disabled  # → enabled (for CAPTCHA protection)
```

### 7. Email Configuration

#### Platform/Supabase (`.env`)
```bash
# SMTP settings for production email
SMTP_ADMIN_EMAIL=admin@example.com  # → admin@your-domain.com
SMTP_HOST=supabase-mail  # → your-smtp-server.com
SMTP_PORT=2500  # → 587 or 465 for SSL
SMTP_USER=fake_mail_user  # → your-smtp-username
SMTP_PASS=fake_mail_password  # → your-smtp-password
SMTP_SENDER_NAME=fake_sender  # → Your Service Name
```

#### Backend (`backend/.env`)
```bash
# Production email service (if using Postmark)
POSTMARK_SENDER_EMAIL=invalid@invalid.com  # → noreply@your-domain.com
```

### 8. Storage & CDN

#### Backend (`backend/.env`)
```bash
# Required for marketplace and file storage
MEDIA_GCS_BUCKET_NAME=  # → your-gcs-bucket-name
```

### 9. Docker Network Configuration

If using Docker in production, ensure services can communicate:

1. Services in `docker-compose.yml` use internal hostnames (e.g., `db`, `redis`, `rabbitmq`)
2. These resolve within Docker networks, but external access requires proper port mapping
3. Update `DB_HOST`, `REDIS_HOST` in backend to use Docker service names if backend runs in Docker

## Deployment Checklist

- [ ] **URLs**: Update all `localhost` references to your actual domain
- [ ] **Security**: Generate new secure keys for all SECRET/KEY variables
- [ ] **Database**: Update database host, credentials, and connection settings
- [ ] **Redis**: Update Redis host (password is optional)
- [ ] **OAuth**: Update all callback URLs for OAuth providers
- [ ] **Email**: Configure production SMTP settings
- [ ] **SSL/TLS**: Use `https://` and `wss://` protocols in production
- [ ] **Ports**: Verify all port configurations match your infrastructure
- [ ] **Environment**: Set `APP_ENV` to `production`
- [ ] **Monitoring**: Enable error tracking (Sentry) and analytics
- [ ] **Storage**: Configure cloud storage for media files

## Quick Start for New Deployment

1. Copy all `.env.default` files to `.env` in their respective directories:
   ```bash
   cp autogpt_platform/.env.default autogpt_platform/.env
   cp autogpt_platform/backend/.env.default autogpt_platform/backend/.env
   cp autogpt_platform/frontend/.env.default autogpt_platform/frontend/.env
   ```

2. Update all URL references from `localhost` to your domain

3. Generate secure keys:
   ```python
   # For ENCRYPTION_KEY and UNSUBSCRIBE_SECRET_KEY
   from cryptography.fernet import Fernet
   print(Fernet.generate_key().decode())
   ```

4. Configure your database and Redis connections

5. Set up OAuth applications and update callback URLs

6. Test connectivity before launching services

## Security Notes

⚠️ **WARNING**: Never commit `.env` files to version control
⚠️ **WARNING**: Always use HTTPS/WSS in production
⚠️ **WARNING**: Rotate keys regularly and use a secrets management system
⚠️ **WARNING**: Default Supabase keys are public - always regenerate for production

## Troubleshooting

### Common Issues:

1. **OAuth callbacks failing**: Ensure `FRONTEND_BASE_URL` matches exactly with OAuth app settings
2. **WebSocket connection errors**: Check that WSS protocol and ports are correctly configured
3. **Database connection timeouts**: Increase `DB_CONNECT_TIMEOUT` for remote databases
4. **CORS errors**: Verify `PLATFORM_BASE_URL` and `FRONTEND_BASE_URL` are correctly set

### Validation Commands:

Test your configuration:
```bash
# Test database connection
psql "postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

# Test Redis connection
redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} ping
# If using password authentication:
# redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} -a ${REDIS_PASSWORD} ping

# Verify Supabase connection
curl ${SUPABASE_URL}/rest/v1/
```

## Additional Resources

- [Supabase Self-Hosting Guide](https://supabase.com/docs/guides/self-hosting)
- [Docker Compose Environment Variables](https://docs.docker.com/compose/environment-variables/)
- Platform-specific READMEs:
  - `/autogpt_platform/README.md` - Platform overview
  - `/autogpt_platform/backend/README.md` - Backend setup
  - `/autogpt_platform/frontend/README.md` - Frontend setup
  - `/autogpt_platform/db/docker/README.md` - Database setup