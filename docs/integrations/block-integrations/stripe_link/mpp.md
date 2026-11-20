# Stripe Link MPP
<!-- MANUAL: file_description -->
Use these blocks to purchase from HTTP endpoints that implement the Machine
Payments Protocol (MPP). The flow reads a merchant's HTTP 402 Stripe challenge,
creates a Shared Payment Token spend request for the user to approve, and then
retries the purchase with that token. The token is not exposed as a graph
output; the pay block sends it to the guarded merchant request only inside an
`Authorization: Payment` credential.
<!-- END MANUAL -->

## Stripe Link Get Payment Challenge

### What it is
MPP step 1 of 3: read a merchant's HTTP 402 payment challenge to learn its network ID and amount. Step 2 is Create Spend Request with credential type 'shared_payment_token' and that network ID; step 3 is MPP Pay. Returns supports_mpp=false for ordinary merchants — use the virtual-card flow for those.

### How it works
<!-- MANUAL: how_it_works -->
The block sends the requested method and JSON body without authentication and
does not follow redirects. A guarded HTTP client rejects disallowed URLs. For a
Stripe `Payment` challenge, the block decodes the bounded base64url request and
returns its network ID, amount, and currency. A successful response without a
402 sets `supports_mpp` to false. A 402 with no challenge or no Stripe challenge
sets `payment_required` to true; a selected Stripe challenge that is malformed
or oversized produces an `error` instead. Other HTTP failures also return an
error because support could not be determined.
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
| supports_mpp | True when the merchant answered 402 with a Stripe payment challenge, so a Shared Payment Token can pay it. False means it cannot be paid this way — check `payment_required` to see whether that is because nothing is owed or because the merchant only accepts a method this block cannot provide. A probe that got no answer at all raises instead, so it is never reported as False. | bool |
| payment_required | True when the merchant demanded payment (HTTP 402) but not via Stripe — an onchain-only merchant, say. With `supports_mpp` false this distinguishes 'pays another way, unreachable from here' from 'served without charging', where the virtual-card flow is the sensible fallback. | bool |
| network_id | Merchant network ID — pass this to Create Spend Request as `network_id` | str |
| amount | Amount the merchant wants, in the smallest currency unit | int |
| currency | Three-letter currency code | str |
| description | What the merchant says the charge is for | str |

### Possible use case
<!-- MANUAL: use_case -->
**Paid API Discovery**: Probe an API endpoint for a Stripe MPP challenge before constructing a payment.

**Spend Request Setup**: Pass `network_id`, `amount`, and `currency` to Create Token Spend Request when `supports_mpp` is true.

**Unsupported Payment Routing**: Stop or choose an advertised payment method when `payment_required` is true but MPP support is false.
<!-- END MANUAL -->

---

## Stripe Link MPP Pay

### What it is
MPP step 3 of 3: spend an approved Shared Payment Token at the merchant's endpoint. Follows Get Payment Challenge (step 1) and Create Token Spend Request (step 2). No card number and no checkout form. The token is single-use, so a failed payment needs a fresh spend request.

### How it works
<!-- MANUAL: how_it_works -->
The block retrieves the spend request from Link and requires both an `approved`
status and a Shared Payment Token. It then probes for the merchant's Stripe
challenge without a Link payment credential and retries once with a generated
`Authorization: Payment` credential. The probe carries caller-supplied headers
other than `Authorization`. Redirects are disabled, and every merchant URL goes
through the guarded HTTP client. `paid` is true only when that
credential-bearing retry was attempted and returned a 2xx response.
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
| paid | True when the merchant accepted the credential-bearing payment request (2xx) | bool |
| response | Merchant's JSON response, e.g. an order or receipt | Dict[str, Any] |

### Possible use case
<!-- MANUAL: use_case -->
**Paid API Purchase**: Submit an approved token spend request to the same merchant endpoint that supplied the challenge.

**Receipt Capture**: Keep the merchant response as the order or receipt only when `paid` is true.

**Failed Payment Recovery**: Create and approve a new spend request after a failed payment instead of reusing the token.
<!-- END MANUAL -->

---
