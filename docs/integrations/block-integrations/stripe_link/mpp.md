# Stripe Link Mpp
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Stripe Link Get Payment Challenge

### What it is
Step 1 of paying an MPP merchant: read its HTTP 402 payment challenge to learn the network ID and amount. Feed those into Create Spend Request with credential type 'shared_payment_token'.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| url | The merchant endpoint to purchase from | str | Yes |
| method | HTTP method the purchase uses | str | No |
| body | JSON body for the purchase request | Dict[str, Any] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message on failure | str |
| supports_mpp | True when the merchant answered 402 with a Stripe payment challenge. False means it is not an MPP merchant and you should use the virtual-card flow instead. | bool |
| network_id | Merchant network ID — pass this to Create Spend Request as `network_id` | str |
| amount | Amount the merchant wants, in the smallest currency unit | int |
| currency | Three-letter currency code | str |
| description | What the merchant says the charge is for | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Stripe Link MPP Pay

### What it is
Step 3 of paying an MPP merchant: spend an approved Shared Payment Token at the merchant's endpoint. No card number and no checkout form. The token is single-use — a failed payment needs a fresh spend request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spend_request_id | An approved spend request created with credential type 'shared_payment_token' | str | Yes |
| url | The merchant endpoint to purchase from | str | Yes |
| method | HTTP method | str | No |
| body | JSON body for the purchase request | Dict[str, Any] | No |
| headers | Extra headers to send to the merchant | Dict[str, str] | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message on failure | str |
| status_code | HTTP status the merchant returned | int |
| paid | True when the merchant accepted the payment (2xx) | bool |
| response | Merchant's JSON response, e.g. an order or receipt | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
