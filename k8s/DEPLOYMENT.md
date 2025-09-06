# AutoGPT Kubernetes Deployment Guide

## Complete Deployment Instructions

This guide provides step-by-step instructions for deploying AutoGPT on Kubernetes using Google Cloud Platform (GKE).

### Prerequisites Checklist

- [ ] Google Cloud Platform account with billing enabled
- [ ] Docker installed and running
- [ ] kubectl installed
- [ ] Helm 3 installed
- [ ] Terraform installed
- [ ] gcloud CLI installed and configured
- [ ] AutoGPT source code cloned

### Step 1: Environment Setup

```bash
# Configure gcloud
gcloud auth login
gcloud config set project your-project-id

# Set environment variables (or update them in configs/.env)
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"  # Change to your preferred region
export AUTOGPT_SOURCE_DIR="/path/to/AutoGPT"
```

### Step 2: Build and Push Docker Images

The AutoGPT platform requires custom Docker images for each service. You have three options for building and storing these images:

#### Option A: Build Locally (Default - Recommended for Testing)
```bash
# Build images locally without pushing to any registry
cd /path/to/AutoGPT-Kubernetes/k8s
export AUTOGPT_SOURCE_DIR=/path/to/AutoGPT
./scripts/build-images.sh
```

#### Option B: Push to Google Artifact Registry (For GCP Users)
```bash
# Build and push to Google Artifact Registry
export REGISTRY_TYPE=gcr
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
./scripts/build-images.sh
```

#### Option C: Push to Docker Hub (For Public Sharing)
```bash
# Build and push to Docker Hub
docker login  # Login to Docker Hub first
export REGISTRY_TYPE=dockerhub
export DOCKER_HUB_ORG=your-dockerhub-username
./scripts/build-images.sh
```

The following images will be built:
- `autogpt-server` - Main API server
- `autogpt-executor` - Agent executor
- `autogpt-websocket` - WebSocket server
- `autogpt-scheduler` - Task scheduler
- `autogpt-notification` - Notification service
- `autogpt-database-manager` - Database manager
- `autogpt-migrate` - Database migration tool
- `autogpt-builder` - Frontend application

### Step 3: Configure Environment Variables

```bash
# Copy default configuration
cp configs/.env.default configs/.env

# Edit configuration
vim configs/.env
```

Update these critical values:
- `AUTOGPT_DOMAIN` - Your domain or use "autogpt.localhost" for testing
- Database passwords (generate secure passwords)
- API keys for LLM providers (OpenAI, Anthropic, etc.)
- Supabase secrets (will be auto-generated if not provided)

### Step 4: Deploy Infrastructure with Terraform

```bash
# Navigate to Terraform directory
cd terraform/gcp

# Initialize Terraform
terraform init

# Review the deployment plan
terraform plan

# Apply the configuration
terraform apply
```

This creates:
- GKE cluster with 3 nodes
- VPC networking
- Cloud NAT for outbound connections
- Static IP addresses for load balancers
- Firewall rules

### Step 5: Deploy Services to Kubernetes

```bash
# Return to k8s directory
cd ../..

# Run the deployment script
./scripts/deploy.sh
```

This deploys:
1. **Infrastructure Services**:
   - Supabase (PostgreSQL + Auth)
   - Redis
   - RabbitMQ

2. **AutoGPT Services**:
   - Database Manager
   - REST API Server
   - WebSocket Server
   - Task Scheduler
   - Notification Service
   - Web UI (Builder)

### Step 6: Verify Deployment

```bash
# Check all pods are running
kubectl get pods -n autogpt

# Check services and load balancers
kubectl get services -n autogpt

# View logs for troubleshooting
kubectl logs -n autogpt -l app=autogpt-server
```

### Step 7: Configure DNS (Production)

For production deployments with a real domain:

1. Get the load balancer IP:
```bash
kubectl get ingress -n autogpt
```

2. Create DNS A records:
- `autogpt.yourdomain.com` → Load Balancer IP
- `api.autogpt.yourdomain.com` → Load Balancer IP
- `ws.autogpt.yourdomain.com` → Load Balancer IP

### Step 8: Local Testing Setup

For local testing without a domain:

```bash
# Run the local access setup script
./scripts/setup-local-access.sh
```

This updates your `/etc/hosts` file to map domains to load balancer IPs.

## Updating Images

When you need to update the AutoGPT services with new code:

```bash
# Pull latest AutoGPT code
cd $AUTOGPT_SOURCE_DIR
git pull origin main

# Rebuild and push images
cd /path/to/AutoGPT-Kubernetes/k8s
./scripts/build-images.sh

# Update deployments
kubectl rollout restart deployment -n autogpt
```

## Monitoring and Maintenance

### View Real-time Logs
```bash
# Stream logs from all services
kubectl logs -n autogpt -f --selector=app=autogpt-server

# View specific pod logs
kubectl logs -n autogpt <pod-name> -f
```

### Scale Services
```bash
# Scale a specific service
kubectl scale deployment autogpt-server -n autogpt --replicas=5

# Enable autoscaling
kubectl autoscale deployment autogpt-server -n autogpt \
  --min=2 --max=10 --cpu-percent=70
```

### Database Backup
```bash
# Create manual backup
kubectl exec -n autogpt supabase-postgresql-0 -- \
  pg_dump -U postgres > backup-$(date +%Y%m%d-%H%M%S).sql

# Schedule automatic backups (add to crontab)
0 2 * * * kubectl exec -n autogpt supabase-postgresql-0 -- pg_dump -U postgres > /backups/backup-$(date +\%Y\%m\%d).sql
```

## Troubleshooting Guide

### Issue: Pods stuck in ImagePullBackOff

**Solution**: Images haven't been built or pushed correctly.
```bash
# Check if images exist in registry
gcloud artifacts docker images list ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/autogpt

# Rebuild and push
./scripts/build-images.sh

# Update image pull secrets if needed
kubectl create secret docker-registry gcr-json-key \
  --docker-server=${GCP_REGION}-docker.pkg.dev \
  --docker-username=_json_key \
  --docker-password="$(cat ~/key.json)" \
  -n autogpt
```

### Issue: Services can't connect to database

**Solution**: Check database status and credentials.
```bash
# Check database pod
kubectl get pod -n autogpt supabase-postgresql-0
kubectl logs -n autogpt supabase-postgresql-0

# Verify secrets
kubectl get secrets -n autogpt
kubectl describe secret autogpt-secrets -n autogpt
```

### Issue: Out of Memory errors

**Solution**: Increase resource limits in Helm values.
```yaml
# Edit helm/autogpt-server/values.yaml
resources:
  limits:
    memory: 4Gi  # Increase this
  requests:
    memory: 2Gi  # And this
```

Then update the deployment:
```bash
helm upgrade autogpt-server ./helm/autogpt-server -n autogpt
```

## Clean Up

To remove the deployment and save costs:

```bash
# Delete Kubernetes resources
kubectl delete namespace autogpt

# Destroy GCP infrastructure
cd terraform/gcp
terraform destroy

# Delete Artifact Registry images (optional)
gcloud artifacts repositories delete autogpt --location=${GCP_REGION}
```

## Next Steps

1. Set up monitoring with Prometheus/Grafana
2. Configure automated backups
3. Implement CI/CD pipeline
4. Set up SSL certificates with cert-manager
5. Configure horizontal pod autoscaling
6. Implement network policies for security

## Support

For issues specific to this Kubernetes deployment:
- Open an issue on GitHub
- Join the AutoGPT Discord

For AutoGPT platform issues:
- See [AutoGPT Documentation](https://docs.agpt.co)
- Join [AutoGPT Discord](https://discord.gg/autogpt)