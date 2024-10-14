output "website_media_bucket_name" {
  description = "The name of the created website media bucket"
  value       = google_storage_bucket.website_artifacts.name
}

output "website_media_bucket_url" {
  description = "The URL of the created website media bucket"
  value       = google_storage_bucket.website_artifacts.url
}
