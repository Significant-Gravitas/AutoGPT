variable "project_id" {
  description = "The ID of the project"
  type        = string
}

variable "region" {
  description = "The region where the bucket will be created"
  type        = string
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
