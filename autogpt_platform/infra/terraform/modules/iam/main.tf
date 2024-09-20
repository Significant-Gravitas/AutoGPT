resource "google_service_account" "service_accounts" {
  for_each     = var.service_accounts
  account_id   = each.key
  display_name = each.value.display_name
  description  = each.value.description
}

# IAM policy binding for Workload Identity
resource "google_service_account_iam_binding" "workload_identity_binding" {
  for_each           = var.workload_identity_bindings
  service_account_id = google_service_account.service_accounts[each.value.service_account_name].name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[${each.value.namespace}/${each.value.ksa_name}]"
  ]
}

# Role bindings grouped by role
resource "google_project_iam_binding" "role_bindings" {
  for_each = var.role_bindings
  project  = var.project_id
  role     = each.key

  members = each.value
}