# AutoGPT Platform - Kubernetes Deployment Guide

Deploy the complete AutoGPT Platform on Google Cloud Platform using Kubernetes.

## 🚀 Prerequisites

- **GCP Account** with billing enabled and project created
- **AutoGPT source code** cloned locally  
- **gcloud CLI** installed and authenticated (`gcloud auth login`)
- **kubectl** (v1.24+)
- **Helm** (v3.8+)
- **Docker** (for building images)
- **Terraform** (for GCP infrastructure)

## 📋 Deployment Process

### Step 1: Setup GCP Infrastructure with Terraform

```bash
# Navigate to Terraform directory
cd terraform/gcp

# Initialize and apply Terraform
terraform init
terraform plan
terraform apply

# Connect to created GKE cluster
gcloud container clusters get-credentials autogpt-cluster --zone us-central1-a --project [YOUR_PROJECT_ID]

# Create namespace
kubectl create namespace autogpt
```

### Step 2: Build Docker Images

Choose your container registry based on your cloud provider:

#### Option A: Google Cloud Artifact Registry (GCP)
```bash
# Setup for Google Cloud
export REGISTRY_TYPE=gcr
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
export AUTOGPT_SOURCE_DIR=/path/to/AutoGPT
./scripts/build-images.sh

# Images pushed to: [GCP_REGION]-docker.pkg.dev/[PROJECT]/autogpt/
```

#### Option B: Docker Hub (Open Source)
```bash
# Setup for Docker Hub
export REGISTRY_TYPE=dockerhub
export DOCKER_HUB_USERNAME=your-username
export AUTOGPT_SOURCE_DIR=/path/to/AutoGPT
./scripts/build-images.sh

# Images pushed to: docker.io/[USERNAME]/autogpt-[service]
```

#### Option C: AWS ECR
```bash
# Setup for AWS ECR
export REGISTRY_TYPE=ecr
export AWS_ACCOUNT_ID=your-account-id
export AWS_REGION=us-west-2
export AUTOGPT_SOURCE_DIR=/path/to/AutoGPT
./scripts/build-images.sh

# Images pushed to: [AWS_ACCOUNT_ID].dkr.ecr.[AWS_REGION].amazonaws.com/autogpt-[service]
```

#### Option D: Local Registry
```bash
# For local development/testing
export REGISTRY_TYPE=local
export AUTOGPT_SOURCE_DIR=/path/to/AutoGPT
./scripts/build-images.sh

# Images tagged locally: localhost:5000/autogpt-[service]
```

### Step 3: Deploy with Helm

```bash
# Deploy all services using Helm
./scripts/deploy.sh

# Run database migrations (required after first deployment)
./scripts/run-migrations.sh

# Or deploy individual services with custom registry:
helm install redis ./helm/redis -n autogpt
helm install rabbitmq ./helm/rabbit-mq -n autogpt  
helm install supabase ./helm/supabase -n autogpt

# Deploy AutoGPT services with your custom registry
helm install autogpt-server ./helm/autogpt-server -n autogpt \
  --set image.repository=[GCP_REGION]-docker.pkg.dev/[YOUR_PROJECT]/autogpt/autogpt-server

helm install autogpt-builder ./helm/autogpt-builder -n autogpt \
  --set image.repository=[GCP_REGION]-docker.pkg.dev/[YOUR_PROJECT]/autogpt/autogpt-builder

helm install autogpt-websocket ./helm/autogpt-websocket -n autogpt \
  --set image.repository=[GCP_REGION]-docker.pkg.dev/[YOUR_PROJECT]/autogpt/autogpt-server
```

### Step 4: Upgrade Services (After Code Changes)

```bash
# Rebuild images with latest code
./scripts/build-images.sh

# Upgrade specific services
helm upgrade autogpt-builder ./helm/autogpt-builder -n autogpt --wait
helm upgrade autogpt-server ./helm/autogpt-server -n autogpt --wait
helm upgrade autogpt-websocket ./helm/autogpt-websocket -n autogpt --wait
```

### Step 5: Local Access via Port Forwarding

```bash
# Start all port forwards for local development
./scripts/local-access.sh

# Or manually:
kubectl port-forward svc/autogpt-builder 3000:3000 -n autogpt &
kubectl port-forward svc/autogpt-server 8006:8006 -n autogpt &  
kubectl port-forward svc/autogpt-websocket 8001:8001 -n autogpt &
kubectl port-forward svc/supabase-kong 8000:8000 -n autogpt &

# Access URLs:
# Frontend: http://localhost:3000
# API: http://localhost:8006
# WebSocket: ws://localhost:8001  
# Supabase: http://localhost:8000
```

**Important**: Restart port forwards after each Helm upgrade/redeploy.

## ✅ Verification

```bash
# Check all pods are running
kubectl get pods -n autogpt

# Test API health  
curl http://localhost:8006/health

# Test frontend access
open http://localhost:3000
```

## 🏗️ Architecture Components

### Core Services
| Service | Purpose | Port | 
|---------|---------|------|
| autogpt-server | Main API | 8006 |
| autogpt-server-executor | Task execution engine | 8002 |
| autogpt-builder | Frontend UI | 3000 |
| autogpt-websocket | Real-time updates | 8001 |
| autogpt-database-manager | Database access for executors | - |
| autogpt-scheduler | Task scheduling | - |
| autogpt-notification | Notification service | - |

### Infrastructure Services  
| Service | Purpose |
|---------|---------|
| PostgreSQL | Primary database |
| Redis | Cache & sessions |
| RabbitMQ | Message queue |
| Supabase Kong | API Gateway |
| Supabase Auth | Authentication service |

## 🔧 Management Commands

### Monitor Deployment
```bash
# Check status
kubectl get pods,svc -n autogpt

# View logs
kubectl logs -f deployment/autogpt-server -n autogpt
```

### Update Images
```bash
# After pulling latest code, rebuild and upgrade:
./scripts/build-images.sh  
helm upgrade autogpt-builder ./helm/autogpt-builder -n autogpt --wait

# Restart port forwards after upgrade
pkill kubectl && ./scripts/local-access.sh
```

## 🧹 Cleanup

```bash
# Remove everything
kubectl delete namespace autogpt

# Or individual services
helm uninstall autogpt-server autogpt-builder autogpt-websocket redis rabbitmq supabase -n autogpt
```