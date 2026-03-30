# Deployment Environment Variables

This guide documents **all environment variables that must be configured** when deploying AutoGPT to a new server or environment. Use this as a checklist to ensure your deployment works correctly.

## Quick Reference: What MUST Change

When deploying to a new server, these variables **must** be updated from their localhost defaults:

| Variable | Location | Default | Purpose |
|----------|----------|---------|---------|
| `SITE_URL` | `.env` | `http://localhost:3000` | Frontend URL for auth redirects |
| `API_EXTERNAL_URL` | `.env` | `http://localhost:8000` | Public Supabase API URL |
| `SUPABASE_PUBLIC_URL` | `.env` | `http://localhost:8000` | Studio dashboard URL |
| `PLATFORM_BASE_URL` | `backend/.env` | `http://localhost:8000` | Backend platform URL |
| `FRONTEND_BASE_URL` | `backend/.env` | `http://localhost:3000` | Frontend URL for webhooks/OAuth |
| `NEXT_PUBLIC_SUPABASE_URL` | `frontend/.env` | `http://localhost:8000` | Client-side Supabase URL |
| `NEXT_PUBLIC_AGPT_SERVER_URL` | `frontend/.env` | `http://localhost:8006/api` | Client-side backend API URL |
| `NEXT_PUBLIC_AGPT_WS_SERVER_URL` | `frontend/.env` | `ws://localhost:8001/ws` | Client-side WebSocket URL |
| `NEXT_PUBLIC_FRONTEND_BASE_URL` | `frontend/.env` | `http://localhost:3000` | Client-side frontend URL |

---

## Configuration Files

AutoGPT uses multiple `.env` files across different components:

```text
autogpt_platform/
├── .env                    # Supabase/infrastructure config
├── backend/
│   ├── .env.default        # Backend defaults (DO NOT EDIT)
│   └── .env                # Your backend overrides
└── frontend/
    ├── .env.default        # Frontend defaults (DO NOT EDIT)
    └── .env                # Your frontend overrides
```

**Loading Order** (later overrides earlier):

1. `*.env.default` - Base defaults
2. `*.env` - Your overrides
3. Docker `environment:` section
4. Shell environment variables

---

## 1. URL Configuration (REQUIRED)

These URLs must be updated to match your deployment domain/IP.

### Root `.env` (Supabase)

```bash
# Auth redirects - where users return after login
SITE_URL=https://your-domain.com:3000

# Public API URL - exposed to clients
API_EXTERNAL_URL=https://your-domain.com:8000

# Studio dashboard URL
SUPABASE_PUBLIC_URL=https://your-domain.com:8000
```

### Backend `.env`

```bash
# Platform URLs for webhooks and OAuth callbacks
PLATFORM_BASE_URL=https://your-domain.com:8000
FRONTEND_BASE_URL=https://your-domain.com:3000

# Internal Supabase URL (use Docker service name if containerized)
SUPABASE_URL=http://kong:8000  # Docker
# SUPABASE_URL=https://your-domain.com:8000  # External
```

### Frontend `.env`

```bash
# Client-side URLs (used in browser)
NEXT_PUBLIC_SUPABASE_URL=https://your-domain.com:8000
NEXT_PUBLIC_AGPT_SERVER_URL=https://your-domain.com:8006/api
NEXT_PUBLIC_AGPT_WS_SERVER_URL=wss://your-domain.com:8001/ws
NEXT_PUBLIC_FRONTEND_BASE_URL=https://your-domain.com:3000
```

!!! warning "HTTPS Note"
    For production, use HTTPS URLs and `wss://` for WebSocket. You'll need a reverse proxy (nginx, Caddy) with SSL certificates.

!!! info "Port Numbers"
    The port numbers shown (`:3000`, `:8000`, `:8001`, `:8006`) are internal Docker service ports. In production with a reverse proxy, your public URLs typically won't include port numbers (e.g., `https://your-domain.com` instead of `https://your-domain.com:3000`). Configure your reverse proxy to route external traffic to the internal service ports.

---

## 2. Security Keys (MUST REGENERATE)

These default values are **public** and **must be changed** for production.

### Root `.env`

```bash
# Database password
POSTGRES_PASSWORD=<generate-strong-password>

# JWT secret for Supabase auth (min 32 chars)
JWT_SECRET=<generate-random-string>

# Supabase keys (regenerate with matching JWT_SECRET)
ANON_KEY=<regenerate>
SERVICE_ROLE_KEY=<regenerate>

# Studio dashboard credentials
DASHBOARD_USERNAME=<your-username>
DASHBOARD_PASSWORD=<strong-password>

# Encryption keys
SECRET_KEY_BASE=<generate-random-string>
VAULT_ENC_KEY=<generate-32-char-key>  # Run: openssl rand -hex 16
```

### Backend `.env`

```bash
# Must match root POSTGRES_PASSWORD
DB_PASS=<same-as-POSTGRES_PASSWORD>

# Must match root SERVICE_ROLE_KEY
SUPABASE_SERVICE_ROLE_KEY=<same-as-SERVICE_ROLE_KEY>

# Must match root JWT_SECRET
JWT_VERIFY_KEY=<same-as-JWT_SECRET>

# Generate new encryption keys
# Run: python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"
ENCRYPTION_KEY=<generated-fernet-key>
UNSUBSCRIBE_SECRET_KEY=<generated-fernet-key>
```

### Generating Keys

```bash
# Generate Fernet encryption key (for ENCRYPTION_KEY, UNSUBSCRIBE_SECRET_KEY)
python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())"

# Generate random string (for JWT_SECRET, SECRET_KEY_BASE)
openssl rand -base64 32

# Generate 32-character key (for VAULT_ENC_KEY)
openssl rand -hex 16

# Generate Supabase keys (requires matching JWT_SECRET)
# Use: https://supabase.com/docs/guides/self-hosting/docker#generate-api-keys
```

---

## 3. Database Configuration

### Root `.env`

```bash
POSTGRES_HOST=db              # Docker service name or external host
POSTGRES_DB=postgres
POSTGRES_PORT=5432
POSTGRES_PASSWORD=<your-password>
```

### Backend `.env`

```bash
DB_USER=postgres
DB_PASS=<your-password>
DB_NAME=postgres
DB_PORT=5432
DB_HOST=localhost             # Default is localhost; use 'db' in Docker
DB_SCHEMA=platform

# Connection pooling
DB_CONNECTION_LIMIT=12
DB_CONNECT_TIMEOUT=60
DB_POOL_TIMEOUT=300

# Full connection URL (auto-constructed from above in .env.default)
# Variable substitution is handled automatically; only override if you need custom parameters
DATABASE_URL="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}?schema=${DB_SCHEMA}"
```

---

## 4. Service Dependencies

### Redis

```bash
REDIS_HOST=redis              # Docker: 'redis', External: hostname/IP
REDIS_PORT=6379
# REDIS_PASSWORD=             # Uncomment if using authentication
```

### RabbitMQ

```bash
RABBITMQ_DEFAULT_USER=<username>
RABBITMQ_DEFAULT_PASS=<strong-password>
# In Docker, host is 'rabbitmq'
```

---

## 5. Default Ports

| Service | Port | Purpose |
|---------|------|---------|
| Frontend | 3000 | Next.js web UI |
| Kong (Supabase API) | 8000 | API gateway |
| WebSocket Server | 8001 | Real-time updates |
| Executor | 8002 | Agent execution |
| Scheduler | 8003 | Scheduled tasks |
| Database Manager | 8005 | DB operations |
| REST Server | 8006 | Main API |
| Notification Server | 8007 | Notifications |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache/queue |
| RabbitMQ | 5672/15672 | Message queue |
| ClamAV | 3310 | Antivirus scanning |

---

## 6. OAuth Callbacks

When configuring OAuth providers, use this callback URL format:

```text
https://your-domain.com/auth/integrations/oauth_callback
# Or with explicit port if not using a reverse proxy:
# https://your-domain.com:3000/auth/integrations/oauth_callback
```

### Supported OAuth Providers

| Provider | Env Variables | Setup URL |
|----------|---------------|-----------|
| GitHub | `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` | [github.com/settings/developers](https://github.com/settings/developers) |
| Google | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | [console.cloud.google.com](https://console.cloud.google.com/apis/credentials) |
| Discord | `DISCORD_CLIENT_ID`, `DISCORD_CLIENT_SECRET` | [discord.com/developers](https://discord.com/developers/applications) |
| Twitter/X | `TWITTER_CLIENT_ID`, `TWITTER_CLIENT_SECRET` | [developer.x.com](https://developer.x.com) |
| Notion | `NOTION_CLIENT_ID`, `NOTION_CLIENT_SECRET` | [developers.notion.com](https://developers.notion.com) |
| Linear | `LINEAR_CLIENT_ID`, `LINEAR_CLIENT_SECRET` | [linear.app/settings/api](https://linear.app/settings/api/applications/new) |
| Reddit | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` | [reddit.com/prefs/apps](https://reddit.com/prefs/apps) |
| Todoist | `TODOIST_CLIENT_ID`, `TODOIST_CLIENT_SECRET` | [developer.todoist.com](https://developer.todoist.com/appconsole.html) |

---

## 7. Optional Services

### AI/LLM Providers

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=
OPEN_ROUTER_API_KEY=
NVIDIA_API_KEY=
```

### Email (SMTP)

```bash
# Supabase auth emails
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=<username>
SMTP_PASS=<password>
SMTP_ADMIN_EMAIL=admin@example.com

# Application emails (Postmark)
POSTMARK_SERVER_API_TOKEN=
POSTMARK_SENDER_EMAIL=noreply@your-domain.com
```

### Payments (Stripe)

```bash
STRIPE_API_KEY=
STRIPE_WEBHOOK_SECRET=
```

### Error Tracking (Sentry)

```bash
SENTRY_DSN=
```

### Analytics (PostHog)

```bash
POSTHOG_API_KEY=
POSTHOG_HOST=https://eu.i.posthog.com

# Frontend
NEXT_PUBLIC_POSTHOG_KEY=
NEXT_PUBLIC_POSTHOG_HOST=https://eu.i.posthog.com
```

---

## 8. Deployment Checklist

Use this checklist when deploying to a new environment:

### Pre-deployment

- [ ] Clone repository and navigate to `autogpt_platform/`
- [ ] Copy all `.env.default` files to `.env`
- [ ] Determine your deployment domain/IP

### URL Configuration

- [ ] Update `SITE_URL` in root `.env`
- [ ] Update `API_EXTERNAL_URL` in root `.env`
- [ ] Update `SUPABASE_PUBLIC_URL` in root `.env`
- [ ] Update `PLATFORM_BASE_URL` in `backend/.env`
- [ ] Update `FRONTEND_BASE_URL` in `backend/.env`
- [ ] Update all `NEXT_PUBLIC_*` URLs in `frontend/.env`

### Security

- [ ] Generate new `POSTGRES_PASSWORD`
- [ ] Generate new `JWT_SECRET` (min 32 chars)
- [ ] Regenerate `ANON_KEY` and `SERVICE_ROLE_KEY`
- [ ] Change `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD`
- [ ] Generate new `ENCRYPTION_KEY` (backend)
- [ ] Generate new `UNSUBSCRIBE_SECRET_KEY` (backend)
- [ ] Update `DB_PASS` to match `POSTGRES_PASSWORD`
- [ ] Update `JWT_VERIFY_KEY` to match `JWT_SECRET`
- [ ] Update `SUPABASE_SERVICE_ROLE_KEY` to match

### Services

- [ ] Configure Redis connection (if external)
- [ ] Configure RabbitMQ credentials
- [ ] Configure SMTP for emails (if needed)

### OAuth (if using integrations)

- [ ] Register OAuth apps with your callback URL
- [ ] Add client IDs and secrets to `backend/.env`

### Post-deployment

- [ ] Run `docker compose up -d --build`
- [ ] Verify frontend loads at your URL
- [ ] Test authentication flow
- [ ] Test WebSocket connection (real-time updates)

---

## 9. Docker vs External Services

### Running Everything in Docker (Default)

The docker-compose files automatically set internal hostnames:

```yaml
# Internal Docker service names (container-to-container communication)
# These are set automatically in docker-compose.platform.yml
DB_HOST: db
REDIS_HOST: redis
RABBITMQ_HOST: rabbitmq
SUPABASE_URL: http://kong:8000
```

### Using External Services

If using managed services (AWS RDS, Redis Cloud, etc.), override in your `.env`:

```bash
# External PostgreSQL
DB_HOST=your-rds-instance.region.rds.amazonaws.com
DB_PORT=5432

# External Redis
REDIS_HOST=your-redis.cache.amazonaws.com
REDIS_PORT=6379
REDIS_PASSWORD=<if-required>

# External Supabase (hosted)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
```

---

## Related Documentation

- [Getting Started](getting-started.md) - Basic setup guide
- [Advanced Setup](advanced_setup.md) - Development configuration
- [OAuth & SSO](integrating/oauth-guide.md) - Integration setup
