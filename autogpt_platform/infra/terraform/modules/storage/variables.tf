variable "project_id" {
  description = "The ID of the project"
  type        = string
}

variable "region" {
  description = "The region where the bucket will be created"
  type        = string
}

variable "website_media_bucket_name" {
  description = "The name of the bucket to create"
  type        = string
  default     = "website-media"
}