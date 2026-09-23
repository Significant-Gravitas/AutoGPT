# Slant3D Webhook
<!-- MANUAL: file_description -->
Blocks for receiving webhook notifications about Slant3D order status updates.
<!-- END MANUAL -->

## Slant3D Order Webhook

### What it is
This block triggers on Slant3D order status updates and outputs the event details, including tracking information when orders are shipped.

### How it works
<!-- MANUAL: how_it_works -->
This block subscribes to order events on a Slant3D platform. New v2 subscriptions require `platform_id="your-platform-id"`; only retained v1 subscriptions may omit it. Registration is serialized per platform to prevent competing callbacks. Each platform has one webhook URL, so an existing callback belonging to another application requires a dedicated platform. V2 deliveries must include `X-Webhook-Signature-256: sha256=<digest>` and a fresh millisecond `X-Webhook-Timestamp`; invalid signatures, platform mismatches, and invalid event types are rejected.

The block outputs order details and available tracking information. Missing carrier codes become empty strings, and dummy deliveries do not trigger workflows. Retained v1 callbacks have no signature scheme and rely on their unguessable callback URL as a bearer credential: keep that URL private. Reconnect with a v2 key and platform ID to migrate to signed delivery.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| platform_id | Required for new v2 subscriptions; retained v1 subscriptions may omit this platform ID. Use a platform without another webhook. | str | No |
| events | Order status events to subscribe to | Events | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if payload processing failed | str |
| payload | The complete webhook payload received from Slant3D | Dict[str, Any] |
| order_id | The ID of the affected order | str |
| status | The new status of the order | str |
| tracking_number | Shipment tracking number, empty before shipment | str |
| carrier_code | Carrier code when supplied by Slant3D, otherwise empty | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Notifications**: Automatically notify customers via email or SMS when their 3D print order ships.

**Order Tracking**: Update your internal systems with shipping information when orders are fulfilled.

**Inventory Management**: Trigger restocking workflows when orders are completed.
<!-- END MANUAL -->

---
