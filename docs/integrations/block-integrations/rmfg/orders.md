# Rmfg Orders
<!-- MANUAL: file_description -->
Blocks that track paid RMFG orders through production and shipping.
<!-- END MANUAL -->

## RMFG Get Order

### What it is
Fetches an RMFG order's status and shipment tracking

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/orders/{id}`: the status (`received`, `in_production`, `ready_for_pickup`, `shipped`, `delivered`, `cancelled` or `refunded`), estimated ship date, line items, a status history, and carrier tracking once shipped.

An unknown ID is reported as `RMFG not_found_error: <message>`. `tracking_url`, `tracking_number` and `estimated_ship_date` are emitted only when RMFG has them, so a graph can branch on their presence; `amount_total_cents` is `0` until the order carries a total.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| order_id | Order ID from Pay Cart or an order.status_changed event | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| order | The full order | Order |
| order_id | Order ID | str |
| status | received, in_production, ready_for_pickup, shipped, delivered, cancelled or refunded | "received" \| "in_production" \| "ready_for_pickup" \| "shipped" \| "delivered" \| "cancelled" \| "refunded" |
| tracking | Carrier, number and link once shipped | OrderTracking |
| tracking_url | Carrier tracking link, once shipped | str |
| tracking_number | Carrier tracking number, once shipped | str |
| estimated_ship_date | Planned ship date, when known | str |
| amount_total_cents | Amount charged, USD cents | int |

### Possible use case
<!-- MANUAL: use_case -->
**Shipping Notification**: When an `order.status_changed` event arrives, load the order and message the customer with the tracking link.

**Delivery Confirmation**: Check for `delivered` status before closing a support ticket.

**Cancellation Handling**: Detect `cancelled` or `refunded` orders and update the internal record.
<!-- END MANUAL -->

---

## RMFG List Orders

### What it is
Lists the RMFG account's manufacturing orders

### How it works
<!-- MANUAL: how_it_works -->
Reads one page of `/v1/orders`, newest first, with `limit` between 1 and 100, and returns `next_cursor` when there are more. Feed the cursor back in to page; an empty `next_cursor` means the last page.

An invalid cursor is rejected by RMFG and surfaced as `RMFG <code>: <message>`. An account with no orders yields an empty `orders` list, no `order` items and an empty cursor.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| limit | Orders per page. | int | No |
| cursor | next_cursor from a previous page; empty for the first page. | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the request failed | str |
| orders | This page of orders | List[Order] |
| order | One order at a time | Order |
| order_ids | IDs in the same order | List[str] |
| next_cursor | Pass back as cursor to fetch the next page; empty on the last page | str |

### Possible use case
<!-- MANUAL: use_case -->
**Weekly Production Report**: Page through all orders and summarise what is in production versus shipped.

**Open Order Dashboard**: List recent orders and filter to those not yet `delivered`.

**Reconciliation**: Match order totals against the accounting system's records.
<!-- END MANUAL -->

---
