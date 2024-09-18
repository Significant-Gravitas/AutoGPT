output "cluster_name" {
  description = "The name of the cluster"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "The endpoint for the cluster"
  value       = google_container_cluster.primary.endpoint
}

output "node_pool_name" {
  description = "The name of the node pool"
  value       = var.enable_autopilot ? null : google_container_cluster.primary.node_pool[0].name
}
