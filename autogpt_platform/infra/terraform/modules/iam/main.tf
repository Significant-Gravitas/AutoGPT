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

resource "google_iam_workload_identity_pool" "pools" {
  for_each = var.workload_identity_pools
  workload_identity_pool_id = each.key
  display_name = each.value.display_name
}

resource "google_iam_workload_identity_pool_provider" "providers" {
  for_each = merge([
    for pool_id, pool in var.workload_identity_pools : {
      for provider_id, provider in pool.providers :
      "${pool_id}/${provider_id}" => merge(provider, {
        pool_id = pool_id
      })
    }
  ]...)

  workload_identity_pool_id = split("/", each.key)[0]
  workload_identity_pool_provider_id = split("/", each.key)[1]

  attribute_mapping = each.value.attribute_mapping
  oidc {
    issuer_uri = each.value.issuer_uri
    allowed_audiences = each.value.allowed_audiences
  }
  attribute_condition = "assertion.repository_owner==\"Significant-Gravitas\""
}