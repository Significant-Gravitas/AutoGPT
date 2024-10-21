project_id      = "agpt-dev"
region          = "us-central1"
zone            = "us-central1-a"
network_name    = "dev-gke-network"
subnet_name     = "dev-gke-subnet"
subnet_cidr     = "10.0.0.0/24"
cluster_name    = "dev-gke-cluster"
node_count      = 3
node_pool_name  = "dev-main-pool"
machine_type    = "e2-highmem-4"
disk_size_gb    = 100
static_ip_names = ["agpt-server-ip", "agpt-builder-ip", "agpt-ws-server-ip", "agpt-market-ip"]


service_accounts = {
  "dev-agpt-server-sa" = {
    display_name = "AutoGPT Dev Server Account"
    description  = "Service account for agpt dev server"
  },
  "dev-agpt-builder-sa" = {
    display_name = "AutoGPT Dev Builder Account"
    description  = "Service account for agpt dev builder"
  },
   "dev-agpt-ws-server-sa" = {
    display_name = "AutoGPT Dev WebSocket Server Account"
    description  = "Service account for agpt dev websocket server"
  },
   "dev-agpt-market-sa" = {
    display_name = "AutoGPT Dev Market Server Account"
    description  = "Service account for agpt dev market server"
  },
  "dev-github-actions-sa" = {
    display_name = "GitHub Actions Dev Service Account"
    description  = "Service account for GitHub Actions deployments to dev"
  }
}

workload_identity_bindings = {
  "dev-agpt-server-workload-identity" = {
    service_account_name = "dev-agpt-server-sa"
    namespace            = "dev-agpt"
    ksa_name             = "dev-agpt-server-sa"
  },
  "dev-agpt-builder-workload-identity" = {
    service_account_name = "dev-agpt-builder-sa"
    namespace            = "dev-agpt"
    ksa_name             = "dev-agpt-builder-sa"
  },
  "dev-agpt-ws-server-workload-identity" = {
    service_account_name = "dev-agpt-ws-server-sa"
    namespace            = "dev-agpt"
    ksa_name             = "dev-agpt-ws-server-sa"
  },
  "dev-agpt-market-workload-identity" = {
    service_account_name = "dev-agpt-market-sa"
    namespace            = "dev-agpt"
    ksa_name             = "dev-agpt-market-sa"
  },
  "dev-github-actions-workload-identity" = {
    service_account_name = "dev-github-actions-sa"
    namespace            = "dev-agpt"
    ksa_name             = "dev-github-actions-sa"
  }
}

role_bindings = {
  "roles/container.developer" = [
    "serviceAccount:dev-agpt-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-builder-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-ws-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-market-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-github-actions-sa@agpt-dev.iam.gserviceaccount.com"
  ],
  "roles/cloudsql.client" = [
    "serviceAccount:dev-agpt-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-builder-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-market-sa@agpt-dev.iam.gserviceaccount.com"
  ],
  "roles/cloudsql.editor" = [
    "serviceAccount:dev-agpt-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-builder-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-market-sa@agpt-dev.iam.gserviceaccount.com"
  ],
  "roles/cloudsql.instanceUser" = [
    "serviceAccount:dev-agpt-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-builder-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-market-sa@agpt-dev.iam.gserviceaccount.com"
  ],
  "roles/iam.workloadIdentityUser" = [
    "serviceAccount:dev-agpt-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-builder-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-ws-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-market-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-github-actions-sa@agpt-dev.iam.gserviceaccount.com"
  ]
  "roles/compute.networkUser" = [
    "serviceAccount:dev-agpt-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-builder-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-ws-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-market-sa@agpt-dev.iam.gserviceaccount.com"
  ],
  "roles/container.hostServiceAgentUser" = [
    "serviceAccount:dev-agpt-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-builder-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-ws-server-sa@agpt-dev.iam.gserviceaccount.com",
    "serviceAccount:dev-agpt-market-sa@agpt-dev.iam.gserviceaccount.com"
  ],
  "roles/artifactregistry.writer" = [
    "serviceAccount:dev-github-actions-sa@agpt-dev.iam.gserviceaccount.com"
  ],
  "roles/container.viewer" = [
    "serviceAccount:dev-github-actions-sa@agpt-dev.iam.gserviceaccount.com"
  ],
  "roles/iam.serviceAccountTokenCreator" = [
    "principalSet://iam.googleapis.com/projects/638488734936/locations/global/workloadIdentityPools/dev-pool/*",
    "serviceAccount:dev-github-actions-sa@agpt-dev.iam.gserviceaccount.com"
  ]
}

pods_ip_cidr_range     = "10.1.0.0/16"
services_ip_cidr_range = "10.2.0.0/20"

public_bucket_names = ["website-artifacts"]
standard_bucket_names = []
bucket_admins = ["gcp-devops-agpt@agpt.co", "gcp-developers@agpt.co"]

workload_identity_pools = {
  "dev-pool" = {
    display_name = "Development Identity Pool"
    providers = {
      "github" = {
        issuer_uri = "https://token.actions.githubusercontent.com"
        attribute_mapping = {
          "google.subject" = "assertion.sub"
          "attribute.repository" = "assertion.repository"
          "attribute.repository_owner" = "assertion.repository_owner"
        }
      }
    }
    service_accounts = {
      "dev-github-actions-sa" = [
        "Significant-Gravitas/AutoGPT"
      ]
    }
  }
}