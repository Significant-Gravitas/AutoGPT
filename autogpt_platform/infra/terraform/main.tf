terraform {
  required_version = ">= 1.9.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }

  backend "gcs" {
    bucket = "agpt-dev-terraform"
    prefix = "terraform/state"
  }

}

provider "google" {
  project = var.project_id
  zone    = var.zone
}

module "static_ips" {
  source = "./modules/static_ip"

  project_id = var.project_id
  ip_names   = var.static_ip_names
  region     = var.region
}

module "networking" {
  source = "./modules/networking"

  project_id             = var.project_id
  region                 = var.region
  network_name           = var.network_name
  subnet_name            = var.subnet_name
  subnet_cidr            = var.subnet_cidr
  pods_ip_cidr_range     = var.pods_ip_cidr_range
  services_ip_cidr_range = var.services_ip_cidr_range
}

module "gke_cluster" {
  source = "./modules/gke_cluster"

  project_id       = var.project_id
  zone             = var.zone
  cluster_name     = var.cluster_name
  node_pool_name   = var.node_pool_name
  node_count       = var.node_count
  machine_type     = var.machine_type
  disk_size_gb     = var.disk_size_gb
  network          = module.networking.network_self_link
  subnetwork       = module.networking.subnet_self_link
  enable_autopilot = var.enable_autopilot
}

module "iam" {
  source = "./modules/iam"

  project_id                 = var.project_id
  service_accounts           = var.service_accounts
  workload_identity_bindings = var.workload_identity_bindings
  role_bindings              = var.role_bindings
  workload_identity_pools    = var.workload_identity_pools
}

module "storage" {
  source = "./modules/storage"

  project_id = var.project_id
  region = var.region
  standard_bucket_names = var.standard_bucket_names
  public_bucket_names = var.public_bucket_names
  bucket_admins = var.bucket_admins
}
