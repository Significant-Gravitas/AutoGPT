# Slant3D Webhook
<!-- MANUAL: file_description -->
Blocks for receiving webhook notifications about Slant3D order status updates.
<!-- END MANUAL -->

## Slant3D Order Webhook

### What it is
This block triggers on Slant3D order status updates and outputs the event details, including tracking information when orders are shipped.

### How it works
<!-- MANUAL: how_it_works -->
This block subscribes to order status events on a Slant3D platform. New subscriptions require an explicit platform_id and verify the timestamped HMAC-SHA256 signature before accepting deliveries. Each platform supports one webhook URL; use a dedicated platform if another application already owns it.

The block outputs order details and available tracking information. Carrier codes are empty when Slant3D does not provide them, and dummy deliveries do not trigger workflows. Existing v1 subscriptions retain their legacy behavior; reconnect with a v2 key and platform ID to migrate.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| platform_id | Slant3D platform ID for this subscription; use a platform without another webhook | str | No |
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
