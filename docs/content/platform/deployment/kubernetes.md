# AutoGPT Platform Kubernetes Deployment

This guide covers deploying AutoGPT Platform on Kubernetes clusters.

## Prerequisites

- **Kubernetes 1.20+** cluster
- **kubectl** configured
- **Helm 3.x** (optional, for easier management)
- **Persistent Volume** support
- **Ingress Controller** (for external access)

## Quick Deploy with Helm

### Add Helm Repository

```bash
helm repo add autogpt https://helm.significant-gravitas.org/autogpt-platform
helm repo update
```

### Install with Default Configuration

```bash
helm install autogpt-platform autogpt/autogpt-platform \
  --namespace autogpt \
  --create-namespace
```

### Custom Configuration

```bash
# Create values.yaml
cat > values.yaml << EOF
backend:
  image:
    repository: ghcr.io/significant-gravitas/autogpt-platform-backend
    tag: latest
  replicas: 2
  resources:
    requests:
      memory: "2Gi"
      cpu: "1"
    limits:
      memory: "4Gi"
      cpu: "2"

frontend:
  image:
    repository: ghcr.io/significant-gravitas/autogpt-platform-frontend
    tag: latest
  replicas: 2

database:
  enabled: true
  persistence:
    size: 20Gi

redis:
  enabled: true
  persistence:
    size: 5Gi

ingress:
  enabled: true
  hostname: autogpt.yourdomain.com
  tls: true
EOF

helm install autogpt-platform autogpt/autogpt-platform \
  --namespace autogpt \
  --create-namespace \
  --values values.yaml
```

## Manual Kubernetes Deployment

### Namespace

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autogpt
```

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: autogpt-config
  namespace: autogpt
data:
  DATABASE_URL: "postgresql://autogpt:password@postgres:5432/autogpt"
  REDIS_HOST: "redis"
  RABBITMQ_HOST: "rabbitmq"
  AGPT_SERVER_URL: "http://backend:8000/api"
```

### Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: autogpt-secrets
  namespace: autogpt
type: Opaque
data:
  database-password: cGFzc3dvcmQ=  # base64 encoded
  redis-password: cGFzc3dvcmQ=
  jwt-secret: eW91ci1qd3Qtc2VjcmV0
```

### PostgreSQL Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: autogpt
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: autogpt
        - name: POSTGRES_USER
          value: autogpt
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: autogpt-secrets
              key: database-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: autogpt
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: autogpt
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

### Redis Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: autogpt
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        args: ["redis-server", "--requirepass", "$(REDIS_PASSWORD)"]
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: autogpt-secrets
              key: redis-password
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: autogpt
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
  namespace: autogpt
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

### RabbitMQ Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rabbitmq
  namespace: autogpt
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rabbitmq
  template:
    metadata:
      labels:
        app: rabbitmq
    spec:
      containers:
      - name: rabbitmq
        image: rabbitmq:3-management
        env:
        - name: RABBITMQ_DEFAULT_USER
          value: autogpt
        - name: RABBITMQ_DEFAULT_PASS
          valueFrom:
            secretKeyRef:
              name: autogpt-secrets
              key: rabbitmq-password
        ports:
        - containerPort: 5672
        - containerPort: 15672
        volumeMounts:
        - name: rabbitmq-storage
          mountPath: /var/lib/rabbitmq
      volumes:
      - name: rabbitmq-storage
        persistentVolumeClaim:
          claimName: rabbitmq-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: rabbitmq
  namespace: autogpt
spec:
  selector:
    app: rabbitmq
  ports:
  - name: amqp
    port: 5672
    targetPort: 5672
  - name: management
    port: 15672
    targetPort: 15672
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rabbitmq-pvc
  namespace: autogpt
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

### Backend Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: autogpt
spec:
  replicas: 2
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: ghcr.io/significant-gravitas/autogpt-platform-backend:latest
        envFrom:
        - configMapRef:
            name: autogpt-config
        env:
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: autogpt-secrets
              key: jwt-secret
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: backend
  namespace: autogpt
spec:
  selector:
    app: backend
  ports:
  - port: 8000
    targetPort: 8000
```

### Frontend Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: autogpt
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: ghcr.io/significant-gravitas/autogpt-platform-frontend:latest
        envFrom:
        - configMapRef:
            name: autogpt-config
        ports:
        - containerPort: 3000
        resources:
          requests:
            memory: "512Mi"
            cpu: "0.5"
          limits:
            memory: "1Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: autogpt
spec:
  selector:
    app: frontend
  ports:
  - port: 3000
    targetPort: 3000
```

### Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autogpt-ingress
  namespace: autogpt
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - autogpt.yourdomain.com
    secretName: autogpt-tls
  rules:
  - host: autogpt.yourdomain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 3000
```

## Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: autogpt
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring and Observability

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: autogpt-backend
  namespace: autogpt
spec:
  selector:
    matchLabels:
      app: backend
  endpoints:
  - port: "8000"
    path: /metrics
```

### Grafana Dashboard ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: autogpt-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  autogpt-platform.json: |
    {
      "dashboard": {
        "title": "AutoGPT Platform",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total{job=\"backend\"}[5m])"
              }
            ]
          }
        ]
      }
    }
```

## Backup Strategy

### Database Backup CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: autogpt
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:15
            command:
            - /bin/bash
            - -c
            - |
              pg_dump -h postgres -U autogpt autogpt > /backup/autogpt-$(date +%Y%m%d).sql
              # Upload to S3 or other storage
              aws s3 cp /backup/autogpt-$(date +%Y%m%d).sql s3://your-backup-bucket/
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: autogpt-secrets
                  key: database-password
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
```

## Security Best Practices

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: autogpt-network-policy
  namespace: autogpt
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  - from:
    - podSelector: {}
  egress:
  - to:
    - podSelector: {}
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### Pod Security Standards

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autogpt
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

## Troubleshooting

### Debug Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: debug
  namespace: autogpt
spec:
  containers:
  - name: debug
    image: nicolaka/netshoot
    command: ["/bin/bash"]
    args: ["-c", "while true; do ping backend; sleep 30; done"]
  restartPolicy: Never
```

### Common Commands

```bash
# Check pod status
kubectl get pods -n autogpt

# View logs
kubectl logs -f deployment/backend -n autogpt

# Access pod shell
kubectl exec -it deployment/backend -n autogpt -- /bin/bash

# Port forward for local access
kubectl port-forward service/frontend 3000:3000 -n autogpt

# Check resource usage
kubectl top pods -n autogpt
```

## Production Checklist

- [ ] TLS certificates configured
- [ ] Resource limits set
- [ ] Persistent volumes configured
- [ ] Backup strategy implemented
- [ ] Monitoring and alerting set up
- [ ] Network policies applied
- [ ] Security contexts configured
- [ ] Horizontal autoscaling enabled
- [ ] Ingress properly configured
- [ ] Database properly secured