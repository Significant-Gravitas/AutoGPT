# Rmfg Carts
<!-- MANUAL: file_description -->
Blocks that build and update an RMFG cart. A cart is a quoted basket with a website link a person can pay from; setting an address and shipping option completes its totals, after which Pay Cart can charge the saved card. Creating a cart does not place an order.
<!-- END MANUAL -->

## RMFG Create Cart

### What it is
Creates an RMFG cart with a website checkout link for a configured design

### How it works
<!-- MANUAL: how_it_works -->
Posts the same basket shape as Create Quote to `/v1/carts`, optionally with `ship_to` and a `shipping_option_id`. RMFG quotes the cart immediately and returns `cart_url`, an unguessable checkout link, plus `totals` with subtotal, shipping, tax (once the address is known) and the amount to be charged. `is_payable` is true only when the cart is open, its quote is ready, and both address and shipping option are set. The cart URL grants access to anyone holding it, so treat it as a secret.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| design_id | Design ID from Analyze Design. | str | Yes |
| quantity | Completed units of the design. Repeated parts in an assembly are multiplied by their instance count automatically. | int | No |
| material_id | Sheet-metal stock for every sheet part, from List Materials. Leave empty for tube-only designs or when configuration sets it. | str | No |
| configuration | Full manufacturing configuration: per-part material, tube profile, finish, powder coat, hole operations, welds and accepted risks. A non-empty material_id above overrides defaults.material_id. | ManufacturingConfiguration | No |
| quantity_options | Up to ten other quantities to price for comparison. | List[int] | No |
| additional_items | Further configured designs to price in the same basket. | List[QuoteItemRequest] | No |
| client_reference_id | Your own reference for this item, echoed back on the result. | str | No |
| ship_to | Delivery address. Needed for shipping options, tax and API payment. | ShipTo | No |
| shipping_option_id | A shipping_options[].id from a quote or cart with the same address. Can be chosen later with Update Cart. | str | No |
| idempotency_key | Stable key for identical retries; defaults to the node execution ID. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| cart | The full cart, including its latest quote | Cart |
| cart_id | Cart ID | str |
| cart_url | Website checkout link; anyone holding it can pay, keep it private | str |
| status | open, checked_out or expired | "open" \| "checked_out" \| "expired" |
| quote_status | Status of the cart's latest quote; only ready carts can be paid | "processing" \| "requires_input" \| "ready" \| "blocked" \| "expired" \| "failed" |
| is_payable | True when the cart is open, quoted ready, and has an address and shipping option | bool |
| totals | Subtotal, shipping, tax and total | CartTotals |
| amount_total_cents | What checkout will charge, in USD cents | int |
| shipping_options | Delivery choices once ship_to is set; pass an id to Update Cart | List[ShippingOption] |
| requirements | Selections or decisions still needed before ordering | List[Requirement] |
| manufacturing_warnings | Advisories from automatic file preparation; they do not block ordering | List[ManufacturingReviewWarning] |
| order_id | Order ID, once the cart has been paid | str |

### Possible use case
<!-- MANUAL: use_case -->
After the customer approves a quote, the agent creates a cart with their address and sends them the `cart_url` to choose delivery, sign in and pay on rmfg.com.
<!-- END MANUAL -->

---

## RMFG Get Cart

### What it is
Fetches an RMFG cart and its latest quote by ID

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/carts/{id}`, including re-quoted totals and the `order_id` once paid. Use it after a person edits the cart on the website, or to check the outcome of a payment that returned `processing`.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| cart_id | Cart ID from Create Cart | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| cart | The full cart, including its latest quote | Cart |
| cart_id | Cart ID | str |
| cart_url | Website checkout link; anyone holding it can pay, keep it private | str |
| status | open, checked_out or expired | "open" \| "checked_out" \| "expired" |
| quote_status | Status of the cart's latest quote; only ready carts can be paid | "processing" \| "requires_input" \| "ready" \| "blocked" \| "expired" \| "failed" |
| is_payable | True when the cart is open, quoted ready, and has an address and shipping option | bool |
| totals | Subtotal, shipping, tax and total | CartTotals |
| amount_total_cents | What checkout will charge, in USD cents | int |
| shipping_options | Delivery choices once ship_to is set; pass an id to Update Cart | List[ShippingOption] |
| requirements | Selections or decisions still needed before ordering | List[Requirement] |
| manufacturing_warnings | Advisories from automatic file preparation; they do not block ordering | List[ManufacturingReviewWarning] |
| order_id | Order ID, once the cart has been paid | str |

### Possible use case
<!-- MANUAL: use_case -->
Pay Cart returned `processing`; a scheduled run reads the cart until its status is `checked_out`, then passes `order_id` to Get Order.
<!-- END MANUAL -->

---

## RMFG Update Cart

### What it is
Updates an open RMFG cart's address, shipping option or items

### How it works
<!-- MANUAL: how_it_works -->
Patches `/v1/carts/{id}` with whichever of `ship_to`, `shipping_option_id` or `items` you set; omitted fields keep their values. The cart re-quotes on every change, so read the returned totals and `quote_status` before paying. The block refuses an update with nothing to change.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| cart_id | Cart ID from Create Cart | str | Yes |
| ship_to | New delivery address; leave empty to keep the current one. | ShipTo | No |
| shipping_option_id | A shipping_options[].id to select; empty keeps the current one. | str | No |
| items | Replacement basket; empty keeps the current items. | List[QuoteItemRequest] | No |
| idempotency_key | Stable key for identical retries; defaults to the node execution ID. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| cart | The full cart, including its latest quote | Cart |
| cart_id | Cart ID | str |
| cart_url | Website checkout link; anyone holding it can pay, keep it private | str |
| status | open, checked_out or expired | "open" \| "checked_out" \| "expired" |
| quote_status | Status of the cart's latest quote; only ready carts can be paid | "processing" \| "requires_input" \| "ready" \| "blocked" \| "expired" \| "failed" |
| is_payable | True when the cart is open, quoted ready, and has an address and shipping option | bool |
| totals | Subtotal, shipping, tax and total | CartTotals |
| amount_total_cents | What checkout will charge, in USD cents | int |
| shipping_options | Delivery choices once ship_to is set; pass an id to Update Cart | List[ShippingOption] |
| requirements | Selections or decisions still needed before ordering | List[Requirement] |
| manufacturing_warnings | Advisories from automatic file preparation; they do not block ordering | List[ManufacturingReviewWarning] |
| order_id | Order ID, once the cart has been paid | str |

### Possible use case
<!-- MANUAL: use_case -->
The agent shows the `shipping_options` from Create Cart, the customer picks UPS Ground, and Update Cart selects it so tax and total are final.
<!-- END MANUAL -->

---
