output "public_bucket_names" {
  description = "The names of the created website artifacts buckets"
  value       = { for k, v in google_storage_bucket.public_buckets : k => v.name }
}

output "public_bucket_urls" {
  description = "The URLs of the created website artifacts buckets"
  value       = { for k, v in google_storage_bucket.public_buckets : k => v.url }
}

output "standard_bucket_names" {
  description = "The names of the created standard buckets"
  value       = { for k, v in google_storage_bucket.standard_buckets : k => v.name }
}

output "standard_bucket_urls" {
  description = "The URLs of the created standard buckets"
  value       = { for k, v in google_storage_bucket.standard_buckets : k => v.url }
}
