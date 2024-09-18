output "ip_addresses" {
  description = "Map of created static IP addresses"
  value       = { for i, ip in google_compute_global_address.static_ip : var.ip_names[i] => ip.address }
}

output "ip_names" {
  description = "List of full names of the created static IP addresses"
  value       = google_compute_global_address.static_ip[*].name
}