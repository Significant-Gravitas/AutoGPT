# AutoGPT Platform - Kubernetes Deployment Guide

Deploy the complete AutoGPT Platform on Kubernetes using streamlined YAML configurations and environment variables.

## 🚀 Prerequisites

- **Kubernetes cluster** (GKE, EKS, local minikube, etc.)
- **AutoGPT source code** cloned locally  
- **kubectl** (v1.24+) configured for your cluster
- **Docker** (for building images)
- **envsubst** (usually included with `gettext` package)

## 📋 Quick Start

### Step 1: Configure Environment Variables

Copy and customize the environment configuration:

```bash
# Copy example configuration
cp configs/.env.example configs/.env

# Edit configuration (IMPORTANT: Change passwords and secrets!)
vim configs/.env
```

Key variables to customize:
- `POSTGRES_PASSWORD`: Strong database password
- `JWT_SECRET`: 32+ character secret for JWT tokens
- `ANON_KEY`, `SERVICE_ROLE_KEY`: Supabase authentication keys
- `REGISTRY_TYPE`: `gcr` for Google Cloud, `local` for local images
- `GCP_PROJECT_ID`, `GCP_REGION`: For GCR registry
- `AUTOGPT_SOURCE_DIR`: Path to your AutoGPT source code
- `NAMESPACE`: Kubernetes namespace (default: `autogpt`)

### Step 2: Build Images (Optional)

```bash
# Build both server and frontend images
./scripts/deploy-complete.sh --build

# Or build manually
./scripts/build-images.sh
```

### Step 3: Deploy Complete Stack

```bash
# Deploy everything: infrastructure + services + migrations
./scripts/deploy-complete.sh
```

This single command:
1. ✅ Creates namespace
2. ✅ Deploys infrastructure (PostgreSQL, Redis, RabbitMQ, Supabase)
3. ✅ Waits for infrastructure readiness
4. ✅ Runs database migrations
5. ✅ Deploys AutoGPT services
6. ✅ Shows deployment status

### Step 4: Access Services

```bash
# Automated port forwarding (recommended)
./scripts/local-access.sh

# Or manually:
kubectl port-forward svc/autogpt-builder 3000:3000 -n autogpt &
kubectl port-forward svc/autogpt-server 8006:8006 -n autogpt &

# Access URLs:
# Frontend: http://localhost:3000
# API: http://localhost:8006
```

## 🏗️ Architecture

### AutoGPT Services
| Service | Purpose | Port | Image |
|---------|---------|------|-------|
| autogpt-server | Main REST API | 8006 | autogpt-server |
| autogpt-server-executor | Task execution engine | 8002 | autogpt-server |
| autogpt-scheduler | Task scheduling | 8003 | autogpt-server |
| autogpt-websocket | Real-time updates | 8001 | autogpt-server |
| autogpt-notification | Notifications | 8007 | autogpt-server |
| autogpt-database-manager | Database management | 8005 | autogpt-server |
| autogpt-builder | Frontend UI | 3000 | autogpt-builder |

### Infrastructure Services  
| Service | Purpose | Port |
|---------|---------|------|
| supabase-postgresql | Primary database | 5432 |
| redis-master | Cache & sessions | 6379 |
| rabbitmq | Message queue | 5672 |
| supabase-kong | API Gateway | 8000 |
| supabase-auth | Authentication (GoTrue) | 9999 |

## 🔧 Configuration Files

- **configs/.env**: Main configuration with all environment variables
- **infrastructure.yaml**: Infrastructure services (DB, Redis, RabbitMQ, Supabase)
- **autogpt-services.yaml**: AutoGPT application services
- **scripts/deploy-complete.sh**: Main deployment script with environment variable substitution

## 📝 Environment Variable System

All YAML files use `${VARIABLE_NAME}` syntax for environment variable substitution via `envsubst`. The deploy script automatically:

1. Loads variables from `configs/.env`
2. Calculates registry paths based on `REGISTRY_TYPE`
3. Substitutes all variables in YAML files
4. Applies the processed configurations

## 🔄 Management Commands

### Check Status
```bash
kubectl get pods,svc -n autogpt
kubectl logs -f deployment/autogpt-server -n autogpt
```

### Update After Code Changes
```bash
# Rebuild images and redeploy
./scripts/deploy-complete.sh --build
```

### Run Migrations Separately
```bash
./scripts/run-migrations.sh
```

### Clean Up
```bash
kubectl delete namespace autogpt
```

## 🛠️ Registry Options

### Google Cloud Registry (GCR)
```bash
# In configs/.env:
REGISTRY_TYPE=gcr
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
ARTIFACT_REPO=autogpt
```

### Local Images
```bash
# In configs/.env:
REGISTRY_TYPE=local
# Images will be tagged as autogpt-server:latest, autogpt-builder:latest
```

### Production Domain Access
```bash
# For production deployments with LoadBalancer/Ingress
./scripts/setup-local-access.sh
# This configures /etc/hosts entries for your domain
```

## 🔍 Troubleshooting

### Common Issues

1. **kubectl not connected**: Ensure kubectl is configured for your cluster first
   ```bash
   # For GKE: gcloud container clusters get-credentials CLUSTER_NAME --zone ZONE
   # For EKS: aws eks update-kubeconfig --name CLUSTER_NAME
   # For local: kubectl config use-context CONTEXT_NAME
   ```
2. **Image pull errors**: Check your registry configuration and authentication
3. **Pod crashes**: Check logs with `kubectl logs -f pod/POD_NAME -n autogpt`
4. **Database connection**: Ensure PostgreSQL is ready before services start
5. **Environment variables**: Verify all required variables are set in configs/.env

### Debug Commands
```bash
# Check environment variable substitution
envsubst < autogpt-services.yaml

# Verify image existence
kubectl describe pod POD_NAME -n autogpt

# Check service endpoints
kubectl get endpoints -n autogpt
```

## 🚨 Security Notes

- **CHANGE ALL DEFAULT PASSWORDS** in configs/.env
- **Use strong JWT secrets** (32+ characters)
- **Secure your cluster** with proper RBAC and network policies
- **Use TLS** for production deployments