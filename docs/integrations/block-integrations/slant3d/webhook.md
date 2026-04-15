# Slant3D Webhook
<!-- MANUAL: file_description -->
Blocks for receiving webhook notifications about Slant3D order status updates.
<!-- END MANUAL -->

## Slant3D Order Webhook

### What it is
This block triggers on Slant3D order status updates and outputs the event details, including tracking information when orders are shipped.

### How it works
<!-- MANUAL: how_it_works -->
This block subscribes to Slant3D webhook events for order status updates. When an order's status changes (e.g., printing, shipped, delivered), Slant3D sends a webhook notification that triggers your workflow.

The payload includes order details and, when applicable, shipping information like tracking numbers and carrier codes for fulfillment tracking.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| events | Order status events to subscribe to | Events | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if payload processing failed | str |
| payload | The complete webhook payload received from Slant3D | Dict[str, Any] |
| order_id | The ID of the affected order | str |
| status | The new status of the order | str |
| tracking_number | The tracking number for the shipment | str |
| carrier_code | The carrier code (e.g., 'usps') | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Notifications**: Automatically notify customers via email or SMS when their 3D print order ships.

**Order Tracking**: Update your internal systems with shipping information when orders are fulfilled.

**Inventory Management**: Trigger restocking workflows when orders are completed.
<!-- END MANUAL -->

---
