# Rmfg Quotes
<!-- MANUAL: file_description -->
Blocks that price a configured design. Quoting runs DFM as well, so one call returns both price and manufacturability. A quote is immutable and never includes tax or a shipping selection; carts add those.
<!-- END MANUAL -->

## RMFG Create Quote

### What it is
Gets an RMFG price and manufacturability findings for a configured design

### How it works
<!-- MANUAL: how_it_works -->
Builds an `items[]` basket from the inputs — one design with a quantity and configuration, plus any `additional_items` — and posts it to `/v1/quotes` with an `Idempotency-Key`. The `material_id` shortcut becomes `defaults.material_id`. Quantity is completed designs; repeated parts are multiplied by their instance count. `quantity_options` prices the same configuration at other quantities for comparison. The block asks RMFG to hold the request briefly and then polls until pricing leaves `processing`. Amounts are integer USD cents. If `ship_to` is given, the quote includes `shipping_options` whose ids a cart can select. `requirements` and `dfm_issues` explain a `requires_input` or `blocked` status.
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
"Quote 10 of these in 5052, and compare 1, 10 and 25." The agent quotes with quantity 10 and `quantity_options` [1, 25], then reports unit prices at each quantity along with any manufacturability findings.
<!-- END MANUAL -->

---

## RMFG Get Quote

### What it is
Fetches an RMFG quote by ID

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/quotes/{id}`, optionally polling until it is no longer `processing`. Useful after a `quote.ready` webhook or when Create Quote was run with `wait_for_ready` off.
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
A large assembly quote timed out during a run; a later run reads the quote by ID and continues from there.
<!-- END MANUAL -->

---
