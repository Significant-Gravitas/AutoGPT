# Stripe Link Spend Request
<!-- MANUAL: file_description -->
_Add a description of this category of blocks._
<!-- END MANUAL -->

## Stripe Link Create Card Spend Request

### What it is
Create a Stripe Link spend request for a one-time virtual card. Self-hosted only; on AutoGPT Cloud use Create Token Spend Request with the MPP blocks instead.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| payment_method_id | ID of the payment method to use (from list payment methods) | str | Yes |
| context | Description of the purchase context (min 100 characters). Shown to the user when they approve the request. | str | Yes |
| amount | Amount in the currency's smallest unit — cents for USD, but whole units for zero-decimal currencies like JPY (max 50000) | int | Yes |
| currency | 3-letter ISO currency code | str | No |
| request_approval | If true, immediately sends a push notification to the user for approval. Otherwise, call request-approval separately. | bool | No |
| test_mode | Use Stripe test mode — no real money moves. A card request yields the 4242… test card; a token request yields a test token. | bool | No |
| line_items | Itemised breakdown shown to the user on the approval sheet. Each item takes `name` (required) plus optional `quantity`, `unit_amount`, `description`, `sku`, `url`, `image_url` and `product_url`. | List[Dict[str, Any]] | No |
| totals | Total lines shown on the approval sheet. Each takes `type`, `display_text` and `amount`. `type` is one of: subtotal, tax, total, items_base_amount, items_discount, discount, fulfillment, shipping, fee, gift_wrap, tip, store_credit. | List[Dict[str, Any]] | No |
| metadata | Arbitrary key/value data stored on the spend request. Max 50 keys; keys <= 40 chars, values <= 500 chars. | Dict[str, str] | No |
| merchant_name | Name of the merchant, shown on the approval sheet. | str | Yes |
| merchant_url | URL of the merchant website | str | Yes |

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

## Stripe Link Create Token Spend Request

### What it is
MPP step 2 of 3: ask the user to authorize a payment to a merchant that answers HTTP 402, and provision a Shared Payment Token for it. Takes the network ID from the Get Payment Challenge block; step 3 is MPP Pay. For an ordinary checkout form, use Create Card Spend Request instead.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| payment_method_id | ID of the payment method to use (from list payment methods) | str | Yes |
| context | Description of the purchase context (min 100 characters). Shown to the user when they approve the request. | str | Yes |
| amount | Amount in the currency's smallest unit — cents for USD, but whole units for zero-decimal currencies like JPY (max 50000) | int | Yes |
| currency | 3-letter ISO currency code | str | No |
| request_approval | If true, immediately sends a push notification to the user for approval. Otherwise, call request-approval separately. | bool | No |
| test_mode | Use Stripe test mode — no real money moves. A card request yields the 4242… test card; a token request yields a test token. | bool | No |
| line_items | Itemised breakdown shown to the user on the approval sheet. Each item takes `name` (required) plus optional `quantity`, `unit_amount`, `description`, `sku`, `url`, `image_url` and `product_url`. | List[Dict[str, Any]] | No |
| totals | Total lines shown on the approval sheet. Each takes `type`, `display_text` and `amount`. `type` is one of: subtotal, tax, total, items_base_amount, items_discount, discount, fulfillment, shipping, fee, gift_wrap, tip, store_credit. | List[Dict[str, Any]] | No |
| metadata | Arbitrary key/value data stored on the spend request. Max 50 keys; keys <= 40 chars, values <= 500 chars. | Dict[str, str] | No |
| network_id | Merchant network ID, read from the merchant's HTTP 402 `WWW-Authenticate: Payment` challenge — see the Get Payment Challenge block. This identifies the merchant in place of merchant_name/merchant_url. | str | Yes |

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

## Stripe Link Get Spend Request Status

### What it is
Check whether a Stripe Link spend request has been approved yet. Poll this after creating a request and before spending, for both the card and the Shared Payment Token flows. If the status is 'requires_action' the payment method needs attention first — keep polling when `auto_resumes` is true, otherwise resolve the action and create a new request.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spend_request_id | ID of the spend request to check (e.g., lsrq_...) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| status | Current status: pending_approval, requires_action, approved, denied, expired. Wait for `approved` before spending. | str |
| next_action_type | Set when status is `requires_action`: what the user must resolve before approval can proceed, e.g. a 3D Secure challenge. Empty otherwise. | str |
| next_action_message | Human-readable explanation of the required action | str |
| next_action_url | Where the user resolves the required action | str |
| auto_resumes | True when the request clears itself once the action is done (3D Secure), so keep polling. False means it needs a fresh spend request. | bool |

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

## Stripe Link Retrieve Card

### What it is
Get the one-time virtual card number and CVC for an approved spend request, to type into a normal checkout form. Both are stored in clear text with the execution record — do not use this where PCI compliance matters. Self-hosted only; on AutoGPT Cloud use the Shared Payment Token flow with the MPP blocks instead.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| spend_request_id | ID of an approved spend request (e.g., lsrq_...) | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| status | Current status of the spend request | str |
| card_number | Virtual card number. Single-use, capped at the approved amount, and expires at `valid_until`. Stored in clear text with the execution record and readable through the execution-results API — do not enable this block where PCI compliance matters. | str |
| card_cvc | Virtual card CVC. Stored in clear text with the execution record, same as `card_number`. Retaining a CVC after authorization is prohibited under PCI DSS 3.2. | str |
| card_exp_month | Card expiry month | int |
| card_exp_year | Card expiry year | int |
| card_brand | Card brand (visa, mastercard, etc.) | str |
| valid_until | ISO timestamp when the virtual card expires | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
