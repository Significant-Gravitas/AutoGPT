# Rmfg Triggers
<!-- MANUAL: file_description -->
Trigger block that starts a graph on RMFG lifecycle events. Adding it to a graph registers a webhook endpoint with RMFG through the API; removing it deletes the endpoint again.
<!-- END MANUAL -->

## RMFG Event Trigger

### What it is
Triggers when an RMFG design, quote, cart or order changes

### How it works
<!-- MANUAL: how_it_works -->
Select which events to subscribe to; the platform registers an endpoint at `/v1/webhook-endpoints` for exactly those events and stores the signing secret RMFG returns once, at creation. A connected RMFG account needs the `webhooks` permission from the approval page; an API key registers directly. Registration fails with a clear message when an event is unknown, when RMFG refuses (a 403 on a connected account says to reconnect and allow the webhooks permission), or when the response lacks an endpoint ID or a string signing secret, so no half-registered hook is ever kept. Removing the block deletes the endpoint again; an already-deleted endpoint is fine.

Each delivery is verified before it reaches the graph: `X-RMFG-Signature` must carry a `v1=` HMAC-SHA256 of `<X-RMFG-Timestamp>.<raw body>` under the stored secret, and a timestamp that is missing, unparseable or more than 300 seconds off is rejected with 403. A body that is not a JSON object or lacks `type` is rejected with 400, and an event type the block does not know triggers nothing. The event body is `{id, type, created_at, data}`; the block emits the type plus the data object's id, object, status and `status_url` (as empty strings when absent), so the next block can fetch the full resource with Get Design, Get DFM Report, Get Quote, Get Cart or Get Order.
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
**Fulfilment Feed**: Subscribe to `cart.checked_out` and `order.status_changed`, fetch the order, and post it to the fulfilment channel.

**Async Analysis**: Upload with waiting off and continue the graph when `design.ready` fires.

**Failure Alerts**: Route `design.failed` and `quote.failed` events to a person with the resource ID.
<!-- END MANUAL -->

---
