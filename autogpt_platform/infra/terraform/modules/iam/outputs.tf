output "service_account_emails" {
  description = "The emails of the created service accounts"
  value       = { for k, v in google_service_account.service_accounts : k => v.email }
}

output "workload_identity_pools" {
  value = google_iam_workload_identity_pool.pools
}

output "workload_identity_providers" {
  value = {
    for k, v in google_iam_workload_identity_pool_provider.providers : k => v.name
  }
}
