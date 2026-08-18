# Stripe Link Spend Request
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Stripe Link Create Spend Request

### What it is
Ask the user to authorize a payment, then provision a single-use credential for it. Pick the credential type by how the merchant takes payment: 'card' (default) gives a virtual card number to type into a normal checkout form; 'shared_payment_token' is for merchants that answer HTTP 402 (Machine Payments Protocol), and needs the network ID from the Get Payment Challenge block. The user approves on their phone before anything is spendable.

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
| line_items | Itemised breakdown shown to the user on the approval sheet. Each item takes `name` (required) plus optional `quantity`, `unit_amount`, `description`, `sku`, `url`, `image_url` and `product_url`. | List[Dict[str, Any]] | No |
| totals | Total lines shown on the approval sheet. Each takes `type`, `display_text` and `amount`. `type` is one of: subtotal, tax, total, items_base_amount, items_discount, discount, fulfillment, shipping, fee, gift_wrap, tip, store_credit. | List[Dict[str, Any]] | No |
| credential_type | What the spend request provisions. `card` (default) yields a one-time virtual card. `shared_payment_token` yields an SPT for merchants speaking the Machine Payments Protocol (HTTP 402), which also needs `network_id`. | str | No |
| network_id | Merchant network ID, required for `shared_payment_token`. Read it from the merchant's HTTP 402 `WWW-Authenticate: Payment` challenge. | str | No |
| metadata | Arbitrary key/value data stored on the spend request. Max 50 keys; keys <= 40 chars, values <= 500 chars. | Dict[str, str] | No |

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
List the cards and bank accounts in the user's Link wallet. Use this first to pick a payment method ID for Create Spend Request.

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
Check a spend request and collect its credential once the user has approved. Returns virtual card details for a 'card' request, or the Shared Payment Token for a 'shared_payment_token' one. If the status is 'requires_action', the payment method needs attention first — poll again when `auto_resumes` is true, otherwise resolve the action and create a new request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spend_request_id | ID of the spend request to retrieve (e.g., lsrq_...) | str | Yes |
| include_card | Include unmasked card details in the response | bool | No |
| include_shared_payment_token | Include the Shared Payment Token, for spend requests created with `credential_type: shared_payment_token`. | bool | No |

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
| shared_payment_token | One-time Shared Payment Token, when the request was created with `credential_type: shared_payment_token`. Empty otherwise. | str |
| next_action_type | Set when status is `requires_action`: what the user must resolve before approval can be requested, e.g. a 3D Secure challenge. Empty otherwise. | str |
| next_action_message | Human-readable explanation of the required action | str |
| next_action_url | Where the user resolves the required action | str |
| auto_resumes | True when the action resolves itself and this block can simply be polled again (3D Secure). False means a new spend request is needed once the user has acted. | bool |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
