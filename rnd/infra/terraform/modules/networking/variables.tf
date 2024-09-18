variable "project_id" {
  description = "The project ID to host the network in"
}

variable "region" {
  description = "The region to host the network in"
}

variable "network_name" {
  description = "The name of the VPC network"
}

variable "subnet_name" {
  description = "The name of the subnet"
}

variable "subnet_cidr" {
  description = "The CIDR range for the subnet"
}

variable "pods_ip_cidr_range" {
  description = "The IP address range for pods"
  default     = "10.1.0.0/16"
}

variable "services_ip_cidr_range" {
  description = "The IP address range for services"
  default     = "10.2.0.0/20"
}
