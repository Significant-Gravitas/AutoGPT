resource "google_compute_network" "vpc_network" {
  name                    = var.network_name
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = var.subnet_name
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc_network.self_link

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_ip_cidr_range
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_ip_cidr_range
  }
}

