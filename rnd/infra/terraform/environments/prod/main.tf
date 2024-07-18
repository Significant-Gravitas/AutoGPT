
provider "google" {
  project = var.project_id
  region  = var.region
}

module "cloudrun" {
  source = "../../modules/cloudrun"

  project_id    = var.project_id
  region        = var.region
  environment   = "prod"
  server_image  = "gcr.io/${var.project_id}/autogpt-server:prod"
  builder_image = "gcr.io/${var.project_id}/autogpt-builder:prod"
}

variable "project_id" {
  description = "The ID of the Google Cloud project"
  type        = string
}

variable "region" {
  description = "The region to deploy the Cloud Run services"
  type        = string
  default     = "us-central1"
}

output "prod_server_url" {
  value = module.cloudrun.server_url
}

output "prod_builder_url" {
  value = module.cloudrun.builder_url
}