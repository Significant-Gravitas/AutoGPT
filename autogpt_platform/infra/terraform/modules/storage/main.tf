resource "google_storage_bucket" "website_artifacts" {
  name          = "${var.project_id}-${var.website_media_bucket_name}"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  cors {
    origin          = ["*"]
    method          = ["GET", "HEAD", "OPTIONS"]
    response_header = ["*"]
    max_age_seconds = 3600
  }

}

# IAM Policy for public access to Cloud Storage buckets
resource "google_storage_bucket_iam_policy" "public_access" {
  bucket      = google_storage_bucket.website_artifacts.name
  policy_data = jsonencode({
    bindings = [
      {
        role    = "roles/storage.objectViewer"
        members = ["allUsers"]
      },
      {
        role    = "roles/storage.admin"
        members = ["group:gcp-devops-agpt@agpt.co", "group:gcp-developers@agpt.co"]
      }
    ]
  })
} 
