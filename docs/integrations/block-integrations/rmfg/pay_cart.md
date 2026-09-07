# Rmfg Pay Cart
<!-- MANUAL: file_description -->
Block that pays an RMFG cart through the API. This charges a real card and starts production, so the platform marks it as a sensitive action that a person approves before it runs.
<!-- END MANUAL -->

## RMFG Pay Cart

### What it is
Pays an RMFG cart with the saved card and places a real production order

### How it works
<!-- MANUAL: how_it_works -->
Posts to `/v1/carts/{id}/pay` with `card_on_file`, which uses the card saved on the RMFG account page, or with a Stripe PaymentMethod id you manage. Preconditions: the cart is open, its quote is ready, and it has a `ship_to` and `shipping_option_id`; the charge is the cart's `totals.amount_total_cents`. A stable `Idempotency-Key` (defaulting to the node execution ID) makes retries safe: repeating a paid cart returns the existing payment rather than charging twice. A 202 means the outcome is not yet known — `payment_status` is `processing` and `checked_out` is false — so re-read the cart instead of paying again. On success the cart becomes `checked_out` and `order_id` points at the new order.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| cart_id | Open cart whose quote is ready and which has a ship_to and shipping option. Its totals.amount_total_cents will be charged. | str | Yes |
| payment_type | card_on_file charges the card saved on the RMFG account page; payment_method charges a Stripe PaymentMethod you manage. | "card_on_file" \| "payment_method" | No |
| payment_method_id | Stripe PaymentMethod id (pm_...) when payment_type is payment_method. | str | No |
| customer_email | Receipt and order emails; defaults to the account email. | str | No |
| customer_phone | Contact number for the order. | str | No |
| idempotency_key | Stable key so a retry never charges twice; defaults to the node execution ID. Use a new key only for a different purchase. | str | No |

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
| payment_status | paid, processing (check the cart again later), failed or refunded | "paid" \| "processing" \| "failed" \| "refunded" |
| checked_out | True once the order exists | bool |

### Possible use case
<!-- MANUAL: use_case -->
With the customer's explicit approval of the cart total, an ordering agent pays the cart and hands the `order_id` to Get Order for tracking updates.
<!-- END MANUAL -->

---
