# Rmfg Triggers
<!-- MANUAL: file_description -->
Trigger block that starts a graph on RMFG lifecycle events. Adding it to a graph registers a webhook endpoint with RMFG through the API; removing it deletes the endpoint again.
<!-- END MANUAL -->

## RMFG Event Trigger

### What it is
Triggers when an RMFG design, quote, cart or order changes

### How it works
<!-- MANUAL: how_it_works -->
Select which events to subscribe to; the platform registers an endpoint at `/v1/webhook-endpoints` for exactly those events and stores the signing secret RMFG returns. Each delivery is verified: the `X-RMFG-Signature` header must match an HMAC-SHA256 of the timestamp and raw body, and stale timestamps are rejected. The event body is `{id, type, created_at, data}`; the block emits the type plus the data object's id, object, status and `status_url`, so the next block can fetch the full resource with Get Design, Get DFM Report, Get Quote, Get Cart or Get Order.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| events | Which RMFG events start this graph | Events | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| payload | The raw event RMFG delivered | Dict[str, Any] |
| event | RMFG event type, e.g. order.status_changed | str |
| event_id | Unique ID of this event | str |
| resource_id | ID of the design, DFM report, quote, cart or order concerned | str |
| resource_type | design, dfm_report, quote, cart or order | str |
| status | The resource's new status, when the event carries one | str |
| status_url | API URL of the resource, for fetching its full state | str |
| created_at | When RMFG emitted the event | str |

### Possible use case
<!-- MANUAL: use_case -->
Subscribe to `order.status_changed` and `cart.checked_out`. When a customer pays on the website the graph fetches the order and posts it to the fulfilment channel, and later relays each shipping update.
<!-- END MANUAL -->

---
