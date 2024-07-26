resource "google_compute_address" "static_ip" {
  count        = length(var.ip_names)
  name         = "${var.project_id}-${var.ip_names[count.index]}"
  region       = var.region
  address_type = "EXTERNAL"
}