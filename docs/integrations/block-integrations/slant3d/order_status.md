# Slant3D Order Status
<!-- MANUAL: file_description -->
Blocks for processing, listing, tracking, and cancelling Slant3D orders.
<!-- END MANUAL -->

## Slant3D Cancel Order

### What it is
Cancel an order before production starts

### How it works
<!-- MANUAL: how_it_works -->
This block requests cancellation of an existing order using its public ID, such as `order_id="SLANT_1234567890"`. It returns the provider's status message, or `Order cancelled` when a successful response omits that optional message.

An invalid ID, an order that cannot be cancelled at its current production stage, or a provider failure produces an error. Check the order status before cancelling; failed cancellation requests are not automatically retried.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| order_id | Slant3D public order ID to cancel | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Cancellation status message | str |

### Possible use case
<!-- MANUAL: use_case -->
**Customer Cancellations**: Allow customers to cancel orders through your interface.

**Error Recovery**: Cancel orders placed with incorrect details or specifications.

**Order Management**: Implement cancellation functionality in order management dashboards.
<!-- END MANUAL -->

---

## Slant3D Get Orders

### What it is
Get all orders for the account

### How it works
<!-- MANUAL: how_it_works -->
This block collects the public IDs of all orders associated with the Slant3D account. It requests pages of 100 orders and stops at the reported final page, an empty page, or a short page when pagination metadata is absent. An empty account returns `orders=[]`.

Provider failures stop pagination and surface as an error instead of returning a partial list. The output contains IDs such as `SLANT_1234567890`; use Tracking to retrieve an order's status and tracking numbers.
<!-- END MANUAL -->

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| orders | Public IDs of all orders for the account | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Order Dashboard**: Build dashboards showing all orders and their current status.

**Sync Operations**: Regularly sync Slant3D orders with your internal order management system.

**Reporting**: Generate reports on order volume and status distribution.
<!-- END MANUAL -->

---

## Slant3D Process Order

### What it is
Submit an approved Slant3D draft to order physical 3D-printed parts for manufacturing and delivery. Uses the order_id from Estimate Order or Estimate Shipping, charges the connected payment method, and starts production. Run only after the customer approves the quoted order.

### How it works
<!-- MANUAL: how_it_works -->
This block processes an approved, uncharged draft identified by its public `order_id`, for example `order_id="SLANT_1234567890"` from Estimate Order or Estimate Shipping. Processing charges the connected Slant3D payment method and submits the order to production. The output is the processed order ID.

The provider rejects invalid IDs and drafts that cannot be processed, including already-processed orders. If a provider or connection failure prevents confirmation, the block reports an error with the draft ID. Check that order's status before retrying because payment may have completed; processing requests are not automatically retried.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| order_id | Uncharged draft order ID from an estimate block | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| order_id | Processed Slant3D public order ID | str |

### Possible use case
<!-- MANUAL: use_case -->
**Approved Quote Fulfillment**: Process the draft returned by a pricing block after a customer confirms the cost and shipping details.

**Scheduled Production**: Submit an approved draft when its planned manufacturing date arrives.

**Purchase Approval Workflow**: Process a prepared draft after an internal purchaser approves its cost.
<!-- END MANUAL -->

---

## Slant3D Tracking

### What it is
Track order status and shipping

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the current status and available tracking numbers for one public order ID, such as `order_id="SLANT_1234567890"`. Before fulfillment, missing shipment information returns `tracking_numbers=[]`.

Invalid or inaccessible order IDs and provider failures surface as errors. Use the returned status and tracking-number list for order-status pages and shipment notifications.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| order_id | Slant3D public order ID to track | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Order status | str |
| tracking_numbers | Shipment tracking numbers | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Order Status Page**: Display current order status to customers on your website.

**Shipping Notifications**: Get tracking numbers to send shipping notifications to customers.

**Customer Support**: Look up order status quickly for customer service inquiries.
<!-- END MANUAL -->

---
