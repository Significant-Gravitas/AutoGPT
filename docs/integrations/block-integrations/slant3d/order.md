# Slant3D Order
<!-- MANUAL: file_description -->
Blocks for managing 3D print orders through Slant3D.
<!-- END MANUAL -->

## Slant3D Create Order

### What it is
Order physical 3D-printed parts from Slant3D for manufacturing and delivery. Accepts multiple STL URLs, workspace attachments, or uploaded file IDs with per-part material and quantity, including complete project part sets. Creates a draft, charges the connected payment method, and submits the parts to production. Use only with real customer shipping details and order approval. For a printing-only quote without shipping or billing details, use Slant3D Slicer first.

### How it works
<!-- MANUAL: how_it_works -->
This block loads STL URLs, workspace attachments, or data URIs through the shared media loader and confirms their upload, creates a draft, then processes it to charge the Slant3D account payment method and start production. Each item can reuse a file_id instead of uploading again. Provide customer shipping details, a positive quantity, and a filament public ID. Legacy color/profile values are accepted only when they identify one available filament.

Set platform_id, or omit it when the account has exactly one enabled platform. The block returns the public order ID for tracking. Use Estimate Order first when the customer needs to approve a quote.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| platform_id | Slant3D platform ID; may be omitted when the account has one enabled platform | str | No |
| order_number | Your custom order reference, stored as orderNumber in Slant3D metadata | str | No |
| customer | Customer shipping details | CustomerDetails | Yes |
| items | Items to print | List[OrderItem] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| order_id | Slant3D public order ID | str |

### Possible use case
<!-- MANUAL: use_case -->
**E-Commerce Integration**: Automatically submit 3D print orders from your online store.

**Custom Product Fulfillment**: Create orders for on-demand 3D printed products.

**Automated Manufacturing**: Trigger print orders based on inventory levels or customer requests.
<!-- END MANUAL -->

---

## Slant3D Estimate Order

### What it is
Quote a 3D-printed parts order including shipping by creating an uncharged Slant3D draft. Accepts multiple STL files or uploaded file IDs with per-part quantities. Requires real customer shipping details and an account payment method; do not invent an address. For printing-only part or project quotes without shipping details, use Slant3D Slicer for each file instead. Returns a draft ID that Process Order can submit after approval.

### How it works
<!-- MANUAL: how_it_works -->
This block creates an uncharged draft order and returns its public order ID together with printing, shipping, and total costs. Provide the same customer and item details as Create Order. Slant3D requires an account payment method to create a draft, but estimating does not charge it or start production.

After the customer approves the quote, pass the returned order_id to Process Order.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| platform_id | Slant3D platform ID; may be omitted when the account has one enabled platform | str | No |
| order_number | Your custom order reference, stored as orderNumber in Slant3D metadata | str | No |
| customer | Customer shipping details | CustomerDetails | Yes |
| items | Items to print | List[OrderItem] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| total_price | Total price in USD | float |
| shipping_cost | Shipping cost in USD | float |
| printing_cost | Printing cost in USD | float |
| order_id | Uncharged draft ID; pass to Process Order to place it | str |

### Possible use case
<!-- MANUAL: use_case -->
**Price Quoting**: Provide customers with accurate pricing before they commit to orders.

**Budget Planning**: Calculate costs for batch orders or production runs.

**Comparison Shopping**: Get estimates to compare with other printing services.
<!-- END MANUAL -->

---

## Slant3D Estimate Shipping

### What it is
Estimate delivery costs for physical 3D-printed parts by creating an uncharged Slant3D order draft. Requires real customer shipping details and an account payment method. For printing-only quotes without an address or billing setup, use Slant3D Slicer instead. Returns shipping cost and the draft ID; does not charge or submit the order.

### How it works
<!-- MANUAL: how_it_works -->
This block creates an uncharged draft using the destination and items, then returns its shipping cost, currency, and public order ID. Slant3D requires an account payment method for drafts, but estimating does not charge it or start production.

Use this to display shipping costs at checkout. Process the returned draft only after the customer approves the order.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| platform_id | Slant3D platform ID; may be omitted when the account has one enabled platform | str | No |
| order_number | Your custom order reference, stored as orderNumber in Slant3D metadata | str | No |
| customer | Customer shipping details | CustomerDetails | Yes |
| items | Items to print | List[OrderItem] | Yes |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| shipping_cost | Estimated shipping cost in USD | float |
| currency_code | Currency code | str |
| order_id | Uncharged draft ID; pass to Process Order to place it | str |

### Possible use case
<!-- MANUAL: use_case -->
**Checkout Display**: Show shipping costs to customers before they complete orders.

**International Pricing**: Calculate shipping for different destinations to optimize pricing.

**Cost Breakdown**: Provide transparent shipping cost breakdowns to customers.
<!-- END MANUAL -->

---
