
# Public Buckets
resource "google_storage_bucket" "public_buckets" {
  for_each      = toset(var.public_bucket_names)
  name          = "${var.project_id}-${each.value}"
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

resource "google_storage_bucket_iam_policy" "public_access" {
  for_each = google_storage_bucket.public_buckets

  bucket      = each.value.name
  policy_data = jsonencode({
    bindings = [
      {
        role    = "roles/storage.objectViewer"
        members = ["allUsers"]
      },
      {
        role    = "roles/storage.admin"
        members = [for admin in var.bucket_admins : "group:${admin}"]
      }
    ]
  })
}

# Standard Buckets, with default permissions
resource "google_storage_bucket" "standard_buckets" {
  for_each      = toset(var.standard_bucket_names)
  name          = "${var.project_id}-${each.value}"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

}

resource "google_storage_bucket_iam_member" "standard_access" {
  for_each = {
    for pair in setproduct(keys(google_storage_bucket.standard_buckets), ["gcp-devops-agpt@agpt.co", "gcp-developers@agpt.co"]) :
    "${pair[0]}-${pair[1]}" => {
      bucket = google_storage_bucket.standard_buckets[pair[0]].name
      member = "group:${pair[1]}"
    }
  }

  bucket = each.value.bucket
  role   = "roles/storage.objectAdmin"
  member = each.value.member
}
