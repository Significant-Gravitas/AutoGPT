resource "google_compute_global_address" "static_ip" {
  count        = length(var.ip_names)
  name         = "${var.project_id}-${var.ip_names[count.index]}"
  address_type = "EXTERNAL"
}