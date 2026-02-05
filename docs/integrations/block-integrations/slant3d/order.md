# Slant3D Order
<!-- MANUAL: file_description -->
Blocks for managing 3D print orders through Slant3D.
<!-- END MANUAL -->

## Slant3D Cancel Order

### What it is
Cancel an existing order

### How it works
<!-- MANUAL: how_it_works -->
This block cancels an existing order in the Slant3D system using the order ID. The cancellation request is sent to the Slant3D API and returns a status message confirming the cancellation.

Orders can only be cancelled before they enter production. Check order status before attempting cancellation.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| order_id | Slant3D order ID to cancel | str | Yes |

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

## Slant3D Create Order

### What it is
Create a new print order

### How it works
<!-- MANUAL: how_it_works -->
This block creates a new 3D print order through the Slant3D API. Provide customer shipping details and a list of items to print (STL files with specifications). Each item includes file URL, quantity, and filament selection.

The block returns the Slant3D order ID which you can use for tracking and status updates.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| order_number | Your custom order number (or leave blank for a random one) | str | No |
| customer | Customer details for where to ship the item | CustomerDetails | Yes |
| items | List of items to print | List[OrderItem] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| order_id | Slant3D order ID | str |

### Possible use case
<!-- MANUAL: use_case -->
**E-Commerce Integration**: Automatically submit 3D print orders from your online store.

**Custom Product Fulfillment**: Create orders for on-demand 3D printed products.

**Automated Manufacturing**: Trigger print orders based on inventory levels or customer requests.
<!-- END MANUAL -->

---

## Slant3D Estimate Order

### What it is
Get order cost estimate

### How it works
<!-- MANUAL: how_it_works -->
This block calculates a cost estimate for a potential order without actually placing it. Provide the same details as a create order request, and receive a breakdown of printing and shipping costs.

Use this for price quotes before customers commit to orders.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| order_number | Your custom order number (or leave blank for a random one) | str | No |
| customer | Customer details for where to ship the item | CustomerDetails | Yes |
| items | List of items to print | List[OrderItem] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| total_price | Total price in USD | float |
| shipping_cost | Shipping cost | float |
| printing_cost | Printing cost | float |

### Possible use case
<!-- MANUAL: use_case -->
**Price Quoting**: Provide customers with accurate pricing before they commit to orders.

**Budget Planning**: Calculate costs for batch orders or production runs.

**Comparison Shopping**: Get estimates to compare with other printing services.
<!-- END MANUAL -->

---

## Slant3D Estimate Shipping

### What it is
Get shipping cost estimate

### How it works
<!-- MANUAL: how_it_works -->
This block calculates shipping costs for a potential order based on the destination and items. It provides an estimate before placing the full order.

Use this to display shipping costs at checkout or calculate delivery options for customers.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| order_number | Your custom order number (or leave blank for a random one) | str | No |
| customer | Customer details for where to ship the item | CustomerDetails | Yes |
| items | List of items to print | List[OrderItem] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| shipping_cost | Estimated shipping cost | float |
| currency_code | Currency code (e.g., 'usd') | str |

### Possible use case
<!-- MANUAL: use_case -->
**Checkout Display**: Show shipping costs to customers before they complete orders.

**International Pricing**: Calculate shipping for different destinations to optimize pricing.

**Cost Breakdown**: Provide transparent shipping cost breakdowns to customers.
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
| orders | List of orders with their details | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Order Dashboard**: Build dashboards showing all orders and their current status.

**Sync Operations**: Regularly sync Slant3D orders with your internal order management system.

**Reporting**: Generate reports on order volume and status distribution.
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
| order_id | Slant3D order ID to track | str | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| status | Order status | str |
| tracking_numbers | List of tracking numbers | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
**Order Status Page**: Display current order status to customers on your website.

**Shipping Notifications**: Get tracking numbers to send shipping notifications to customers.

**Customer Support**: Look up order status quickly for customer service inquiries.
<!-- END MANUAL -->

---
