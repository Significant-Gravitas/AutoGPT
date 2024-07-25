resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.zone

  dynamic "node_pool" {
    for_each = var.enable_autopilot ? [] : [1]
    content {
      name       = var.node_pool_name
      node_count = var.node_count

      node_config {
        machine_type = var.machine_type
        disk_size_gb = var.disk_size_gb
      }
    }
  }

  network    = var.network
  subnetwork = var.subnetwork
}

