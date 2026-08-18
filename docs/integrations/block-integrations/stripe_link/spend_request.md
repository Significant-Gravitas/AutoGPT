# Stripe Link Spend Request
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Stripe Link Create Spend Request

### What it is
Create a Stripe Link spend request for a one-time payment credential

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| payment_method_id | ID of the payment method to use (from list payment methods) | str | Yes |
| merchant_name | Name of the merchant for this purchase | str | Yes |
| merchant_url | URL of the merchant website | str | Yes |
| context | Description of the purchase context (min 100 characters). Shown to the user when they approve the request. | str | Yes |
| amount | Amount in cents (max 50000) | int | Yes |
| currency | 3-letter ISO currency code | str | No |
| request_approval | If true, immediately sends a push notification to the user for approval. Otherwise, call request-approval separately. | bool | No |
| test_mode | Use test mode (fake card 4242424242424242) | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| spend_request_id | ID of the created spend request | str |
| status | Status: created, pending_approval, approved, denied, etc. | str |
| approval_url | URL the user can visit to approve (if not using push) | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Stripe Link List Payment Methods

### What it is
List payment methods from a Stripe Link wallet

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| payment_methods | List of payment methods in the Link wallet | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Stripe Link Retrieve Spend Request

### What it is
Retrieve a Stripe Link spend request and card credentials

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spend_request_id | ID of the spend request to retrieve (e.g., lsrq_...) | str | Yes |
| include_card | Include unmasked card details in the response | bool | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| status | Current status of the spend request | str |
| card_number | Virtual card number (only if approved and include_card=True) | str |
| card_cvc | Virtual card CVC | str |
| card_exp_month | Card expiry month | int |
| card_exp_year | Card expiry year | int |
| card_brand | Card brand (visa, mastercard, etc.) | str |
| valid_until | ISO timestamp when the virtual card expires | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
