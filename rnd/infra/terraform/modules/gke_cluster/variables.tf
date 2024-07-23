variable "project_id" {
  description = "The project ID to host the cluster in"
}

variable "zone" {
  description = "The zone to host the cluster in"
}

variable "cluster_name" {
  description = "The name for the GKE cluster"
}

variable "node_count" {
  description = "Number of nodes in the cluster"
}

variable "node_pool_name" {
  description = "Name of the node pool in the cluster"
}

variable "machine_type" {
  description = "Type of machine to use for nodes"
}

variable "disk_size_gb" {
  description = "Size of the disk attached to each node, specified in GB"
  default     = 100
}

variable "network" {
  description = "The VPC network to host the cluster in"
}

variable "subnetwork" {
  description = "The subnetwork to host the cluster in"
}

variable "enable_autopilot" {
  description = "Enable Autopilot for this cluster"
  type        = bool
}