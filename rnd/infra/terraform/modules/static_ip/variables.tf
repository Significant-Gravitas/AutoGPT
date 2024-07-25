variable "project_id" {
  description = "The project ID to prepend to IP names"
  type        = string
}

variable "ip_names" {
  description = "List of custom names for static IPs"
  type        = list(string)
}

variable "region" {
  description = "Region to create the static IPs in"
  type        = string
}
