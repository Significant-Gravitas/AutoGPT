variable "project_id" {
  description = "The ID of the Google Cloud project"
  type        = string
}

variable "region" {
  description = "The region to deploy the Cloud Run services"
  type        = string
}

variable "environment" {
  description = "The environment (e.g. dev or prod)"
  type        = string
}

variable "server_image" {
  description = "The Docker image for the server"
  type        = string
}

variable "builder_image" {
  description = "The Docker image for the builder"
  type        = string
}

# Cloud Run service for the server
resource "google_cloud_run_service" "server" {
  name     = "autogpt-server-${var.environment}"
  location = var.region

  template {
    spec {
      containers {
        image = var.server_image

        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }

        env {
          name  = "PORT"
          value = "8000"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Cloud Run service for the builder
resource "google_cloud_run_service" "builder" {
  name     = "autogpt-builder-${var.environment}"
  location = var.region

  template {
    spec {
      containers {
        image = var.builder_image

        resources {
          limits = {
            cpu    = "1000m"
            memory = "512Mi"
          }
        }

        env {
          name  = "PORT"
          value = "3000"
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# IAM policy to make the services public
data "google_iam_policy" "noauth" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
  }
}

# Apply the IAM policy to the server service
resource "google_cloud_run_service_iam_policy" "server_noauth" {
  location = google_cloud_run_service.server.location
  project  = google_cloud_run_service.server.project
  service  = google_cloud_run_service.server.name

  policy_data = data.google_iam_policy.noauth.policy_data
}

# Apply the IAM policy to the builder service
resource "google_cloud_run_service_iam_policy" "builder_noauth" {
  location = google_cloud_run_service.builder.location
  project  = google_cloud_run_service.builder.project
  service  = google_cloud_run_service.builder.name

  policy_data = data.google_iam_policy.noauth.policy_data
}

output "server_url" {
  value = google_cloud_run_service.server.status[0].url
}

output "builder_url" {
  value = google_cloud_run_service.builder.status[0].url
}