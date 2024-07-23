variable "project_id" {
  description = "The project ID to host the cluster in"
}

variable "region" {
  description = "Project region"
  default     = "us-central1"
}

variable "zone" {
  description = "The zone to host the cluster in"
  default     = "us-central1-a"
}

variable "network_name" {
  description = "The name of the VPC network"
  default     = "gke-network"
}

variable "subnet_name" {
  description = "The name of the subnet"
  default     = "gke-subnet"
}

variable "subnet_cidr" {
  description = "The CIDR range for the subnet"
  default     = "10.0.0.0/24"
}

variable "cluster_name" {
  description = "The name for the GKE cluster"
  default     = "gke-cluster"
}

variable "node_count" {
  description = "Number of nodes in the cluster"
  default     = 3
}

variable "node_pool_name" {
  description = "The name for the node pool"
  default     = "default-pool"
}

variable "machine_type" {
  description = "Type of machine to use for nodes"
  default     = "e2-medium"
}

variable "disk_size_gb" {
  description = "Size of the disk attached to each node, specified in GB"
  default     = 100
}

variable "enable_autopilot" {
  description = "Enable Autopilot for this cluster"
  type        = bool
  default     = false
}