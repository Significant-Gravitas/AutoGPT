terraform {
  required_version = ">= 1.9.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

provider "google" {
  project = var.project_id
  zone    = var.zone
}

# Configure Google client
data "google_client_config" "default" {}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "container.googleapis.com",
    "compute.googleapis.com",
    "sqladmin.googleapis.com",
    "servicenetworking.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com"
  ])
  
  service = each.value
  disable_on_destroy = false
}

# Static IP addresses
resource "google_compute_global_address" "static_ips" {
  for_each = toset(var.static_ip_names)
  name     = "${var.cluster_name}-${each.value}"
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = var.network_name
  auto_create_subnetworks = false
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = var.subnet_name
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc.id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_ip_cidr_range
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_ip_cidr_range
  }

  private_ip_google_access = true
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name

  # Enable Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Enable network policy
  network_policy {
    enabled = true
  }

  # IP allocation policy
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Master authorized networks
  master_authorized_networks_config {
    cidr_blocks {
      cidr_block   = "0.0.0.0/0"
      display_name = "All"
    }
  }

  # Enable private cluster
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  depends_on = [google_project_service.required_apis]
}

# Main node pool
resource "google_container_node_pool" "primary_nodes" {
  name       = var.node_pool_name
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = var.min_node_count

  autoscaling {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }

  node_config {
    preemptible  = false
    machine_type = var.machine_type

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      env = var.environment
    }

    tags = ["gke-node", "${var.cluster_name}-gke"]
    metadata = {
      disable-legacy-endpoints = "true"
    }

    disk_size_gb = var.disk_size_gb
    disk_type    = "pd-standard"
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Stateful node pool for databases and stateful services
resource "google_container_node_pool" "stateful_nodes" {
  count      = var.enable_stateful_pool ? 1 : 0
  name       = "${var.node_pool_name}-stateful"
  location   = var.zone
  cluster    = google_container_cluster.primary.name
  node_count = var.stateful_node_count

  node_config {
    preemptible  = false
    machine_type = var.stateful_machine_type

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      env  = var.environment
      type = "stateful"
    }

    taint {
      key    = "stateful"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    tags = ["gke-stateful-node", "${var.cluster_name}-gke"]
    metadata = {
      disable-legacy-endpoints = "true"
    }

    disk_size_gb = var.stateful_disk_size_gb
    disk_type    = "pd-ssd"
  }

  management {
    auto_repair  = true
    auto_upgrade = true # Required when release channel is set
  }
}

# Service accounts
resource "google_service_account" "gke_nodes" {
  account_id   = "${var.cluster_name}-nodes"
  display_name = "GKE Node Service Account"
}

resource "google_service_account" "service_accounts" {
  for_each = var.service_accounts

  account_id   = each.key
  display_name = each.value.display_name
  description  = each.value.description
}

# Workload Identity bindings
resource "google_service_account_iam_binding" "workload_identity" {
  for_each = var.workload_identity_bindings

  service_account_id = "projects/${var.project_id}/serviceAccounts/${each.value.service_account_name}@${var.project_id}.iam.gserviceaccount.com"
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[${each.value.namespace}/${each.value.ksa_name}]",
  ]

  depends_on = [google_service_account.service_accounts]
}

# IAM role bindings
resource "google_project_iam_binding" "role_bindings" {
  for_each = var.role_bindings

  project = var.project_id
  role    = each.key
  members = each.value

  depends_on = [google_service_account.service_accounts]
}

# Note: Database will be deployed via Helm chart (PostgreSQL + Supabase)
# No CloudSQL needed for simplified deployment

# GCS Buckets
resource "google_storage_bucket" "public_buckets" {
  for_each = toset(var.public_bucket_names)
  
  name     = each.value
  location = var.region

  uniform_bucket_level_access = false

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "standard_buckets" {
  for_each = toset(var.standard_bucket_names)
  
  name     = each.value
  location = var.region

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }
}