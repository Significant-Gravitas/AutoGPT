# Slant3D Order Webhook

### What it is
This block triggers on Slant3D order status updates and outputs the event details, including tracking information when orders are shipped.

### What it does
This block triggers on Slant3D order status updates and outputs the event details, including tracking information when orders are shipped.

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Inputs
| Input | Description | Type | Required |
|-------|-------------|------|----------|
| events | Order status events to subscribe to | Events | No |

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if payload processing failed | str |
| payload | The complete webhook payload received from Slant3D | Dict[str, True] |
| order_id | The ID of the affected order | str |
| status | The new status of the order | str |
| tracking_number | The tracking number for the shipment | str |
| carrier_code | The carrier code (e.g., 'usps') | str |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---
