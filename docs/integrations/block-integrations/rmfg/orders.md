# Rmfg Orders
<!-- MANUAL: file_description -->
Blocks that track paid RMFG orders through production and shipping.
<!-- END MANUAL -->

## RMFG Get Order

### What it is
Fetches an RMFG order's status and shipment tracking

### How it works
<!-- MANUAL: how_it_works -->
Fetches `/v1/orders/{id}`: the status (`received`, `in_production`, `ready_for_pickup`, `shipped`, `delivered`, `cancelled` or `refunded`), estimated ship date, line items, a status history, and carrier tracking once shipped. `tracking_url` and `tracking_number` are only emitted when they exist.
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
| tracking_number | Carrier tracking number | str |
| estimated_ship_date | Planned ship date | str |
| amount_total_cents | Amount charged, USD cents | int |

### Possible use case
<!-- MANUAL: use_case -->
An `order.status_changed` webhook starts the graph; Get Order loads the details and the agent messages the customer with the tracking link.
<!-- END MANUAL -->

---

## RMFG List Orders

### What it is
Lists the RMFG account's manufacturing orders

### How it works
<!-- MANUAL: how_it_works -->
Reads one page of `/v1/orders`, newest first, and returns `next_cursor` when there are more. Feed the cursor back in to page; an empty cursor means the last page.
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
A weekly report agent pages through all orders and summarises what is in production versus shipped.
<!-- END MANUAL -->

---
