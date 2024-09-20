output "service_account_emails" {
  description = "The emails of the created service accounts"
  value       = { for k, v in google_service_account.service_accounts : k => v.email }
}