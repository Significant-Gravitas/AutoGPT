variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "network_name" {
  description = "Name of the VPC network"
  type        = string
  default     = "autogpt-network"
}

variable "subnet_name" {
  description = "Name of the subnet"
  type        = string
  default     = "autogpt-subnet"
}

variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "autogpt-cluster"
}

variable "node_pool_name" {
  description = "Name of the node pool"
  type        = string
  default     = "main-pool"
}

variable "node_count" {
  description = "Number of nodes in the node pool"
  type        = number
  default     = 2
}

variable "min_node_count" {
  description = "Minimum number of nodes in the node pool"
  type        = number
  default     = 2
}

variable "max_node_count" {
  description = "Maximum number of nodes in the node pool"
  type        = number
  default     = 20
}

variable "machine_type" {
  description = "Machine type for the nodes"
  type        = string
  default     = "e2-highmem-4"
}

variable "disk_size_gb" {
  description = "Disk size in GB for each node"
  type        = number
  default     = 100
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "enable_autopilot" {
  description = "Enable GKE Autopilot mode"
  type        = bool
  default     = false
}

variable "enable_stateful_pool" {
  description = "Enable stateful node pool for databases"
  type        = bool
  default     = true
}

variable "stateful_node_count" {
  description = "Number of stateful nodes"
  type        = number
  default     = 1
}

variable "stateful_machine_type" {
  description = "Machine type for stateful nodes"
  type        = string
  default     = "e2-standard-4"
}

variable "stateful_disk_size_gb" {
  description = "Disk size for stateful nodes"
  type        = number
  default     = 100
}

variable "pods_ip_cidr_range" {
  description = "IP CIDR range for pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "services_ip_cidr_range" {
  description = "IP CIDR range for services"
  type        = string
  default     = "10.2.0.0/20"
}

variable "static_ip_names" {
  description = "Names for static IP addresses"
  type        = list(string)
  default     = ["server-ip", "builder-ip", "websocket-ip"]
}

variable "service_accounts" {
  description = "Service accounts to create"
  type = map(object({
    display_name = string
    description  = string
  }))
  default = {
    "autogpt-server-sa" = {
      display_name = "AutoGPT Server Service Account"
      description  = "Service account for AutoGPT server"
    },
    "autogpt-builder-sa" = {
      display_name = "AutoGPT Builder Service Account"
      description  = "Service account for AutoGPT builder"
    },
    "autogpt-websocket-sa" = {
      display_name = "AutoGPT WebSocket Service Account"
      description  = "Service account for AutoGPT WebSocket server"
    }
  }
}

variable "workload_identity_bindings" {
  description = "Workload identity bindings"
  type = map(object({
    service_account_name = string
    namespace            = string
    ksa_name             = string
  }))
  default = {
    "autogpt-server-wi" = {
      service_account_name = "autogpt-server-sa"
      namespace            = "autogpt"
      ksa_name             = "autogpt-server-sa"
    }
  }
}

variable "role_bindings" {
  description = "IAM role bindings"
  type        = map(list(string))
  default     = {}
}

variable "public_bucket_names" {
  description = "Names of public GCS buckets to create"
  type        = list(string)
  default     = []
}

variable "standard_bucket_names" {
  description = "Names of standard GCS buckets to create"
  type        = list(string)
  default     = []
}

variable "bucket_admins" {
  description = "List of users/groups with admin access to buckets"
  type        = list(string)
  default     = []
}

variable "database_instance_name" {
  description = "Name of the CloudSQL instance"
  type        = string
  default     = "autogpt-postgres"
}

variable "database_name" {
  description = "Name of the database"
  type        = string
  default     = "autogpt"
}

variable "database_user" {
  description = "Database username"
  type        = string
  default     = "autogpt"
}

variable "database_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "instance_tier" {
  description = "CloudSQL instance tier"
  type        = string
  default     = "db-custom-4-16384" # 4 vCPU, 16GB RAM
}