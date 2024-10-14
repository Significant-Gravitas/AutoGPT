output "bucket_name" {
  description = "The name of the created bucket"
  value       = google_storage_bucket.website_artifacts.name
}

output "bucket_url" {
  description = "The URL of the created bucket"
  value       = google_storage_bucket.website_artifacts.url
}
