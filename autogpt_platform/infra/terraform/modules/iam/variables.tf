variable "project_id" {
  description = "The ID of the project"
  type        = string
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

variable "workload_identity_pools" {
  type = map(object({
    display_name = string
    providers = map(object({
      issuer_uri = string
      attribute_mapping = map(string)
      allowed_audiences = optional(list(string))
    }))
    service_accounts = map(list(string))  # Map of SA to list of allowed principals
  }))
  default = {}
}