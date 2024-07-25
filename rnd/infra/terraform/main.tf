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

  project_id   = var.project_id
  region       = var.region
  network_name = var.network_name
  subnet_name  = var.subnet_name
  subnet_cidr  = var.subnet_cidr
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
