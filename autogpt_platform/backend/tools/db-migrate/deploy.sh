#!/bin/bash
#
# Build and deploy db-migrate tool to GKE
#
# Usage:
#   ./deploy.sh [command] [args...]
#
# Examples:
#   ./deploy.sh solo --user-id abc-123
#   ./deploy.sh quick
#   ./deploy.sh full
#
# Environment variables required:
#   SOURCE_URL - Source database URL (Supabase)
#   DEST_URL   - Destination database URL (GCP Cloud SQL)
#
# Optional:
#   PROJECT_ID - GCP project (default: agpt-dev)
#   NAMESPACE  - K8s namespace (default: dev-agpt)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env file if it exists
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Check required env vars
if [[ -z "${SOURCE_URL:-}" ]]; then
    echo "ERROR: SOURCE_URL environment variable is required"
    echo "Set it or create a .env file in this directory"
    exit 1
fi

if [[ -z "${DEST_URL:-}" ]]; then
    echo "ERROR: DEST_URL environment variable is required"
    echo "Set it or create a .env file in this directory"
    exit 1
fi

PROJECT_ID="${PROJECT_ID:-agpt-dev}"
REGION="${REGION:-us-central1}"
IMAGE="gcr.io/${PROJECT_ID}/db-migrate:latest"
NAMESPACE="${NAMESPACE:-dev-agpt}"

echo "=== Building db-migrate ==="
cd "$SCRIPT_DIR"
docker build --platform linux/amd64 -t "$IMAGE" .

echo ""
echo "=== Pushing to GCR ==="
docker push "$IMAGE"

echo ""
echo "=== Deploying to GKE ==="

# Get the command and args
CMD="${1:-quick}"
shift || true
ARGS="${*:-}"

# Create a unique job name
JOB_NAME="db-migrate-$(date +%s)"
SECRET_NAME="db-migrate-creds-$(date +%s)"

# Create k8s secret for database credentials (won't be visible in job spec)
echo "Creating secret: ${SECRET_NAME}"
kubectl -n "${NAMESPACE}" create secret generic "${SECRET_NAME}" \
    --from-literal=SOURCE_URL="${SOURCE_URL}" \
    --from-literal=DEST_URL="${DEST_URL}" \
    --dry-run=client -o yaml | kubectl apply -f -

cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: ${NAMESPACE}
spec:
  ttlSecondsAfterFinished: 3600
  backoffLimit: 0
  template:
    spec:
      serviceAccountName: dev-agpt-server-sa
      restartPolicy: Never
      containers:
      - name: migrate
        image: ${IMAGE}
        args: ["${CMD}"${ARGS:+, $(echo "$ARGS" | sed 's/[^ ]*/"\0"/g' | tr ' ' ',')}]
        env:
        - name: SOURCE_URL
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: SOURCE_URL
        - name: DEST_URL
          valueFrom:
            secretKeyRef:
              name: ${SECRET_NAME}
              key: DEST_URL
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
EOF

echo ""
echo "=== Job created: ${JOB_NAME} ==="
echo ""
echo "View logs:"
echo "  kubectl -n ${NAMESPACE} logs -f job/${JOB_NAME}"
echo ""
echo "Delete job:"
echo "  kubectl -n ${NAMESPACE} delete job ${JOB_NAME}"
