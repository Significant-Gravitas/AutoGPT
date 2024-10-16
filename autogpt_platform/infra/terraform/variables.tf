variable "project_id" {
  description = "The project ID to host the cluster in"
  type        = string
}

variable "region" {
  description = "Project region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The zone to host the cluster in"
  type        = string
  default     = "us-central1-a"
}

variable "network_name" {
  description = "The name of the VPC network"
  type        = string
  default     = "gke-network"
}

variable "subnet_name" {
  description = "The name of the subnet"
  type        = string
  default     = "gke-subnet"
}

variable "subnet_cidr" {
  description = "The CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "cluster_name" {
  description = "The name for the GKE cluster"
  type        = string
  default     = "gke-cluster"
}

variable "node_count" {
  description = "Number of nodes in the cluster"
  type        = number
  default     = 3
}

variable "node_pool_name" {
  description = "The name for the node pool"
  type        = string
  default     = "default-pool"
}

variable "machine_type" {
  description = "Type of machine to use for nodes"
  type        = string
  default     = "e2-medium"
}

variable "disk_size_gb" {
  description = "Size of the disk attached to each node, specified in GB"
  type        = number
  default     = 100
}

variable "enable_autopilot" {
  description = "Enable Autopilot for this cluster"
  type        = bool
  default     = false
}

variable "static_ip_names" {
  description = "List of custom names for static IPs"
  type        = list(string)
  default     = ["ip-1", "ip-2", "ip-3"]
}

variable "service_accounts" {
  description = "Map of service accounts to create"
  type = map(object({
    display_name = string
    description  = string
  }))
  default = {}
}

variable "workload_identity_bindings" {
  description = "Map of Workload Identity bindings to create"
  type = map(object({
    service_account_name = string
    namespace            = string
    ksa_name             = string
  }))
  default = {}
}

variable "role_bindings" {
  description = "Map of roles to list of members"
  type        = map(list(string))
  default     = {}
}

variable "pods_ip_cidr_range" {
  description = "The IP address range for pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "services_ip_cidr_range" {
  description = "The IP address range for services"
  type        = string
  default     = "10.2.0.0/20"
}

variable "public_bucket_names" {
  description = "List of bucket names that should be publicly accessible"
  type        = list(string)
  default     = []
}

variable "standard_bucket_names" {
  description = "List of bucket names that should be publicly accessible"
  type        = list(string)
  default     = []
}

variable "bucket_admins" {
  description = "List of groups that should be admins of the buckets"
  type        = list(string)
  default     = ["gcp-devops-agpt@agpt.co", "gcp-developers@agpt.co"]
}

variable "workload_identity_pools" {
  type = map(object({
    display_name = string
    providers = map(object({
      issuer_uri = string
      attribute_mapping = map(string)
      allowed_audiences = optional(list(string))
    }))
    service_accounts = map(list(string))
  }))
  default = {}
  description = "Configuration for workload identity pools and their providers"
}



