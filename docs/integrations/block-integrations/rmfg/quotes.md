# Rmfg Quotes
<!-- MANUAL: file_description -->
Blocks that price a configured design. Quoting runs DFM as well, so one call returns both price and manufacturability. A quote is immutable and never includes tax or a shipping selection; carts add those.
<!-- END MANUAL -->

## RMFG Create Quote

### What it is
Gets a live price from RMFG, a manufacturer that laser-cuts, bends and ships real sheet-metal and tube parts, plus manufacturability findings. Quote with the stocked material closest to the part's detected thickness and report any mismatch; quote exactly the quantity the customer asked for, and every unique part of a project at its required quantity; a requires_input result means a selection is missing, blocked means a finding must be resolved or accepted

### How it works
<!-- MANUAL: how_it_works -->
Builds an `items[]` basket from the inputs (one design with a quantity and configuration, plus any `additional_items`) and posts it to `/v1/quotes` with an `Idempotency-Key`. The `material_id` shortcut becomes `defaults.material_id`. `quantity` is completed designs and must be at least 1, and `quantity_options` takes at most ten entries, each at least 1; both are checked before any request is sent. RMFG holds the connection briefly and the block then polls until pricing leaves `processing`, bounded by `timeout_seconds`. Amounts are integer USD cents, reported as `0` until priced; with `ship_to` set the quote also lists `shipping_options` a cart can select.

`requires_input` and `blocked` are statuses, not errors: `requirements` says what is missing and `dfm_issues` what blocks manufacture, so the graph can fix the configuration and quote again. A quote RMFG marks `failed` comes back with that status and its error object on `quote`; HTTP errors are reported as `RMFG <code>: <message>` and an exhausted wait as `Timed out after Ns waiting for quote <id>`.
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
| ship_to | Destination, to include delivery options in the quote. | ShipTo | No |
| wait_for_ready | Poll until pricing finishes instead of returning at once. | bool | No |
| timeout_seconds | How long to wait for pricing when wait_for_ready is on. | int | No |
| idempotency_key | Stable key for identical retries; defaults to the node execution ID. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| quote | The full quote | Quote |
| quote_id | Quote ID | str |
| status | processing, requires_input, ready, blocked, expired or failed | "processing" \| "requires_input" \| "ready" \| "blocked" \| "expired" \| "failed" |
| is_ready | True when the quote can be ordered | bool |
| amount_total_cents | Total in USD cents (shipping and tax excluded until a cart) | int |
| amount_subtotal_cents | Sum of items in USD cents | int |
| unit_amount_cents | Price per completed unit of the first design, in USD cents | int |
| items | Per-design pricing, line items and DFM findings | List[QuotedDesign] |
| quantity_options | Prices at the other quantities requested, for the first design | List[QuantityOption] |
| shipping_options | Delivery choices when ship_to was given; pick an id for the cart | List[ShippingOption] |
| requirements | Selections or decisions still needed before ordering | List[Requirement] |
| dfm_issues | Manufacturability findings across every design | List[DFMIssue] |

### Possible use case
<!-- MANUAL: use_case -->
**Quantity Comparison**: Quote 10 units with `quantity_options` [1, 25] and report the unit price at each quantity.

**Multi-Part Project**: Price every unique part of an assembly in one basket, each at its required quantity.

**Delivery Estimate**: Include `ship_to` to get shipping options and lead times alongside the price.
<!-- END MANUAL -->

---

## RMFG Get Quote

### What it is
Fetches an RMFG quote by ID

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/quotes/{id}` and, with `wait_for_ready` on, polls until the quote is no longer `processing`, bounded by `timeout_seconds`. Outputs match Create Quote.

An unknown ID is reported as `RMFG not_found_error: <message>`; a quote past its `expires_at` returns status `expired` and should be re-created rather than ordered. Amount outputs are `0` while the quote is still processing.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| quote_id | Quote ID from Create Quote | str | Yes |
| wait_for_ready | Poll until pricing finishes instead of returning at once. | bool | No |
| timeout_seconds | How long to wait for pricing when wait_for_ready is on. | int | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| quote | The full quote | Quote |
| quote_id | Quote ID | str |
| status | processing, requires_input, ready, blocked, expired or failed | "processing" \| "requires_input" \| "ready" \| "blocked" \| "expired" \| "failed" |
| is_ready | True when the quote can be ordered | bool |
| amount_total_cents | Total in USD cents (shipping and tax excluded until a cart) | int |
| amount_subtotal_cents | Sum of items in USD cents | int |
| unit_amount_cents | Price per completed unit of the first design, in USD cents | int |
| items | Per-design pricing, line items and DFM findings | List[QuotedDesign] |
| quantity_options | Prices at the other quantities requested, for the first design | List[QuantityOption] |
| shipping_options | Delivery choices when ship_to was given; pick an id for the cart | List[ShippingOption] |
| requirements | Selections or decisions still needed before ordering | List[Requirement] |
| dfm_issues | Manufacturability findings across every design | List[DFMIssue] |

### Possible use case
<!-- MANUAL: use_case -->
**Resume a Long Quote**: Read a large assembly quote by ID after an earlier run timed out.

**Webhook Follow-Up**: Load the finished quote when a `quote.ready` event arrives.

**Expiry Check**: Confirm a saved quote is still `ready` before turning it into a cart.
<!-- END MANUAL -->

---
