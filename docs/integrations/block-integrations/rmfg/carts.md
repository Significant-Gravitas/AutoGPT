# Rmfg Carts
<!-- MANUAL: file_description -->
Blocks that build and update an RMFG cart. A cart is a quoted basket with a website link a person can pay from; setting an address and shipping option completes its totals, after which Pay Cart can charge the saved card. Creating a cart does not place an order.
<!-- END MANUAL -->

## RMFG Create Cart

### What it is
Creates an RMFG cart with a website checkout link for one or more configured designs, priced live with shipping and tax once an address is set. A cart is not an order; the person pays on the link or Pay Cart charges the saved card after approval

### How it works
<!-- MANUAL: how_it_works -->
Posts the same basket shape as Create Quote to `/v1/carts`, optionally with `ship_to` and a `shipping_option_id`, under an `Idempotency-Key`; `quantity` and `quantity_options` are validated as for quotes. RMFG quotes the cart immediately and returns `cart_url`, an unguessable checkout link, plus `totals` with subtotal, shipping and tax (once the address is known) and the amount that will be charged. `is_payable` is true only when the cart is open, its quote is `ready`, and both address and shipping option are set.

A `shipping_option_id` that does not belong to the given address is rejected by RMFG and surfaced as `RMFG <code>: <message>`; a missing material shows up as `quote_status` `requires_input` with `requirements` rather than as an error. `order_id` is emitted only after the cart has been paid. The cart URL grants access to anyone holding it, so treat it as a secret.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| design_id | Design ID from Analyze Design. | str | Yes |
| quantity | Completed units of the design: exactly the number the customer asked for (one part means 1). Repeated parts in an assembly are multiplied by their instance count automatically. | int | No |
| material_id | Sheet-metal stock for every sheet part, from List Materials. Required for a price: pick the stocked thickness closest to the part's detected thickness and mention the difference. Leave empty only for tube-only designs or when configuration sets it. | str | No |
| configuration | Full manufacturing configuration: per-part material, tube profile, finish, powder coat, hole operations, welds and accepted risks. A non-empty material_id above overrides defaults.material_id. | ManufacturingConfiguration | No |
| quantity_options | Up to ten other quantities to price alongside quantity, only when the customer wants a comparison; the main quantity stays as requested. | List[int] | No |
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
| order_id | Order ID; only emitted once the cart has been paid | str |

### Possible use case
<!-- MANUAL: use_case -->
**Website Checkout Hand-Off**: Create a cart with the customer's address and send them `cart_url` to pick delivery, sign in and pay.

**Agent Ordering Prep**: Build a cart with address and shipping option so Pay Cart can charge the saved card after approval.

**Tax-Inclusive Total**: Show a customer the final total including shipping and tax before they commit.
<!-- END MANUAL -->

---

## RMFG Get Cart

### What it is
Fetches an RMFG cart and its latest quote by ID

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/carts/{id}`, including re-quoted totals, the current `status` (`open`, `checked_out` or `expired`) and the `order_id` once paid. Use it after a person edits the cart on the website, or to check the outcome of a payment that returned `processing`.

An unknown ID is reported as `RMFG not_found_error: <message>`. `order_id` is only emitted for a paid cart, so a graph can branch on its presence, and `is_payable` is recomputed on every read.
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
| order_id | Order ID; only emitted once the cart has been paid | str |

### Possible use case
<!-- MANUAL: use_case -->
**Payment Settlement**: Poll a cart after a `processing` payment until it is `checked_out`, then pass `order_id` to Get Order.

**Customer Edits**: Re-read a cart the customer changed on rmfg.com before quoting the final total back to them.

**Expiry Guard**: Check a cart is still `open` before asking for approval to pay it.
<!-- END MANUAL -->

---

## RMFG Update Cart

### What it is
Updates an open RMFG cart's address, shipping option or items

### How it works
<!-- MANUAL: how_it_works -->
Patches `/v1/carts/{id}` with whichever of `ship_to`, `shipping_option_id` or `items` you set; omitted fields keep their current values, and an empty `items` list keeps the current basket rather than emptying it. The cart re-quotes on every change, so read the returned totals and `quote_status` before paying.

The block refuses to run with nothing to change (`Nothing to update: set ship_to, shipping_option_id or items`). RMFG rejects updates to a `checked_out` or `expired` cart and shipping options that do not match the address; both are surfaced as `RMFG <code>: <message>`.
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
| order_id | Order ID; only emitted once the cart has been paid | str |

### Possible use case
<!-- MANUAL: use_case -->
**Shipping Selection**: Let the customer pick from `shipping_options`, then select it so tax and total are final.

**Address Correction**: Replace a mistyped delivery address and re-read the re-quoted totals.

**Quantity Change**: Replace `items` with the same design at a new quantity after the customer changes their order.
<!-- END MANUAL -->

---
