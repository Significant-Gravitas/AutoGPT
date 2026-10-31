# Stripe Link Spend Request
<!-- MANUAL: file_description -->
Use these blocks to choose a saved Link payment method, create a spend request,
and wait for the user's decision. Shared Payment Token requests work with the
MPP blocks on every deployment. The virtual-card create and retrieve blocks are
self-hosted only because Retrieve Card emits a PAN and CVC that are persisted
with the execution, while allowing Create Card without retrieval would leave an
unusable flow.
<!-- END MANUAL -->

## Stripe Link Create Card Spend Request

### What it is
Create a Stripe Link spend request for a one-time virtual card. Self-hosted only; on AutoGPT Cloud use Create Token Spend Request with the MPP blocks instead.

### How it works
<!-- MANUAL: how_it_works -->
The block posts the selected payment-method ID, merchant name and URL, purchase
context, amount, currency, test-mode setting, and optional approval-sheet fields
to Link. The context must be at least 100 characters, the amount must be from 1
through 50,000 in the currency's smallest unit, and the amount on any supplied
`totals` entry whose type is `total` must equal that amount. When
`request_approval` is true, the create request also asks Link to start approval;
the block returns the resulting status and any approval URL. It does not
retrieve the card. Validation, processing, or Link request failures emit
`error`; callers must not continue to Retrieve Card after a failure.
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
**Standard Checkout**: Request a virtual card for a merchant that accepts payment through a conventional checkout form.

**Approval-Gated Purchase**: Show exact purchase details, then pass the approved request to Retrieve Card.

**Self-Hosted Automation**: Use this flow only where persisting the eventual card number and CVC is acceptable.
<!-- END MANUAL -->

---

## Stripe Link Create Token Spend Request

### What it is
MPP step 2 of 3: ask the user to authorize a payment to a merchant that answers HTTP 402, and provision a Shared Payment Token for it. Takes the network ID from the Get Payment Challenge block; step 3 is MPP Pay. For an ordinary checkout form, use Create Card Spend Request instead.

### How it works
<!-- MANUAL: how_it_works -->
The block posts the same approval and amount fields as the card request, but
identifies the merchant with the nonblank `network_id` from its HTTP 402
challenge and sets `credential_type` to `shared_payment_token`. It does not
send a merchant name or URL. When `request_approval` is true, Link starts the
approval flow as part of creation; use the returned spend-request ID to poll
for the result.
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
**Paid API Access**: Authorize an MPP-protected API call without exposing a card number.

**Challenge-Based Request**: Carry `network_id`, `amount`, and `currency` from Get Payment Challenge into the approval flow.

**Token Payment Handoff**: Pass the spend-request ID to MPP Pay only after its status becomes `approved`.
<!-- END MANUAL -->

---

## Stripe Link Get Spend Request Status

### What it is
Check whether a Stripe Link spend request has been approved yet. Poll this after creating a request and before spending, for both the card and the Shared Payment Token flows. If the status is 'requires_action' the payment method needs attention first — keep polling when `auto_resumes` is true, otherwise resolve the action and create a new request.

### How it works
<!-- MANUAL: how_it_works -->
The block validates the `lsrq_...` ID and retrieves that spend request from
Link. On a successful retrieval it returns the current status. A failed Link
request yields only `error`. For `requires_action`, a successful response also
provides the action type, message, URL, and whether Link will resume the request
automatically. Only an explicit `new_spend_request` resolution sets
`auto_resumes` to false; missing or unfamiliar resolutions keep polling to
avoid creating a second request while the first can still resume.
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
**Approval Polling**: Wait for `approved` before continuing to MPP Pay or Retrieve Card.

**Required-Action Recovery**: Send the user to `next_action_url` when present or surface the action message and type otherwise.

**Safe Retry Routing**: Keep polling when `auto_resumes` is true and create a new request only when it is false.
<!-- END MANUAL -->

---

## Stripe Link List Payment Methods

### What it is
List the cards and bank accounts in the user's Link wallet. Use this first to pick a payment method ID for Create Spend Request.

### How it works
<!-- MANUAL: how_it_works -->
The block makes an authenticated `GET /payment-details` request and projects
each result onto a fixed set of selection fields: ID, type, name, default flag,
and, for cards, brand, last four digits, and expiry. It does not pass through a
card number, CVC, or unknown fields that Link may add to its response.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| payment_methods | List of payment methods in the Link wallet | List[Dict[str, Any]] |

### Possible use case
<!-- MANUAL: use_case -->
**Default Method Selection**: Choose the saved default payment method when it matches the user's intent.

**Explicit User Choice**: Present method names and masked card details when the user should select one.

**Spend Request Wiring**: Pass the chosen method's `id` into a create block as `payment_method_id`.
<!-- END MANUAL -->

---

## Stripe Link Retrieve Card

### What it is
Get the one-time virtual card number and CVC for an approved spend request, to type into a normal checkout form. Both are stored in clear text with the execution record — do not use this where PCI compliance matters. Self-hosted only; on AutoGPT Cloud use the Shared Payment Token flow with the MPP blocks instead.

### How it works
<!-- MANUAL: how_it_works -->
This self-hosted-only block validates the `lsrq_...` ID and requests the spend
request with its card data included. It returns the status first and emits card
fields only when the status is `approved`; denied, expired, or otherwise
unapproved requests produce an error without emitting a number or CVC. When
emitted, those values are ordinary block outputs and remain in the execution
record.
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
**Conventional Checkout**: Complete a standard payment form with the card from an approved spend request.

**Approved-Only Retrieval**: Stop when the request is denied or expired instead of trying old card details.

**Sensitive-Data Boundary**: Keep the card number and CVC inside an operator-controlled environment designed to handle them.
<!-- END MANUAL -->

---
