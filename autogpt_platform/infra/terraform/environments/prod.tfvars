project_id      = "agpt-prod"
region          = "us-central1"
zone            = "us-central1-a"
network_name    = "prod-gke-network"
subnet_name     = "prod-gke-subnet"
subnet_cidr     = "10.0.0.0/24"
cluster_name    = "prod-gke-cluster"
node_count      = 4
node_pool_name  = "prod-main-pool"
machine_type    = "e2-highmem-4"
disk_size_gb    = 100
static_ip_names = ["agpt-backend-ip", "agpt-frontend-ip", "agpt-ws-backend-ip", "agpt-market-ip"]


service_accounts = {
  "prod-agpt-backend-sa" = {
    display_name = "AutoGPT prod backend Account"
    description  = "Service account for agpt prod backend"
  },
  "prod-agpt-frontend-sa" = {
    display_name = "AutoGPT prod frontend Account"
    description  = "Service account for agpt prod frontend"
  },
   "prod-agpt-ws-backend-sa" = {
    display_name = "AutoGPT prod WebSocket backend Account"
    description  = "Service account for agpt prod websocket backend"
  },
   "prod-agpt-market-sa" = {
    display_name = "AutoGPT prod Market backend Account"
    description  = "Service account for agpt prod market backend"
  }
}

workload_identity_bindings = {
  "prod-agpt-backend-workload-identity" = {
    service_account_name = "prod-agpt-backend-sa"
    namespace            = "prod-agpt"
    ksa_name             = "prod-agpt-backend-sa"
  },
  "prod-agpt-frontend-workload-identity" = {
    service_account_name = "prod-agpt-frontend-sa"
    namespace            = "prod-agpt"
    ksa_name             = "prod-agpt-frontend-sa"
  },
  "prod-agpt-ws-backend-workload-identity" = {
    service_account_name = "prod-agpt-ws-backend-sa"
    namespace            = "prod-agpt"
    ksa_name             = "prod-agpt-ws-backend-sa"
  },
  "prod-agpt-market-workload-identity" = {
    service_account_name = "prod-agpt-market-sa"
    namespace            = "prod-agpt"
    ksa_name             = "prod-agpt-market-sa"
  }
}

role_bindings = {
  "roles/container.developer" = [
    "serviceAccount:prod-agpt-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-frontend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-ws-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-market-sa@agpt-prod.iam.gserviceaccount.com"
  ],
  "roles/cloudsql.client" = [
    "serviceAccount:prod-agpt-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-frontend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-market-sa@agpt-prod.iam.gserviceaccount.com"
  ],
  "roles/cloudsql.editor" = [
    "serviceAccount:prod-agpt-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-frontend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-market-sa@agpt-prod.iam.gserviceaccount.com"
  ],
  "roles/cloudsql.instanceUser" = [
    "serviceAccount:prod-agpt-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-frontend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-market-sa@agpt-prod.iam.gserviceaccount.com"
  ],
  "roles/iam.workloadIdentityUser" = [
    "serviceAccount:prod-agpt-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-frontend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-ws-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-market-sa@agpt-prod.iam.gserviceaccount.com"
  ]
  "roles/compute.networkUser" = [
    "serviceAccount:prod-agpt-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-frontend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-ws-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-market-sa@agpt-prod.iam.gserviceaccount.com"
  ],
  "roles/container.hostServiceAgentUser" = [
    "serviceAccount:prod-agpt-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-frontend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-ws-backend-sa@agpt-prod.iam.gserviceaccount.com",
    "serviceAccount:prod-agpt-market-sa@agpt-prod.iam.gserviceaccount.com"
  ]
}

pods_ip_cidr_range     = "10.1.0.0/16"
services_ip_cidr_range = "10.2.0.0/20"

public_bucket_names = ["website-artifacts"]
standard_bucket_names = []
bucket_admins = ["gcp-devops-agpt@agpt.co", "gcp-developers@agpt.co"]