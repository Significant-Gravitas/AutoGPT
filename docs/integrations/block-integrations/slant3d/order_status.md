# Slant3D Order Status
<!-- MANUAL: file_description -->
Blocks for processing, listing, tracking, and cancelling Slant3D orders.
<!-- END MANUAL -->

## Slant3D Cancel Order

### What it is
Cancel an order before production starts

### How it works
<!-- MANUAL: how_it_works -->
This block cancels an existing order in the Slant3D system using the order ID. The cancellation request is sent to the Slant3D API and returns a status message confirming the cancellation.

Orders can only be cancelled before they enter production. Check order status before attempting cancellation.
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
This block retrieves all orders associated with your Slant3D account. It returns a list of orders with their current status and details.

Use this for order management dashboards or to sync order data with your systems.
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
This block processes an existing draft identified by its public order ID. Processing charges the Slant3D account payment method and submits the order to production.

Use the order_id returned by Estimate Order or Estimate Shipping after the customer approves the quote. The block returns the processed order ID and status.
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
<!-- END MANUAL -->

---

## Slant3D Tracking

### What it is
Track order status and shipping

### How it works
<!-- MANUAL: how_it_works -->
This block retrieves the current status and shipping tracking information for a specific order. It returns the order status and any available tracking numbers.

Use this to provide customers with real-time order status updates.
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
