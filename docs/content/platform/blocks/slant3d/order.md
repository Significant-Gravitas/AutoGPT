# Slant3D Cancel Order

### What it is
Cancel an existing order.

### What it does
Cancel an existing order

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Slant3D Create Order

### What it is
Create a new print order.

### What it does
Create a new print order

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Slant3D Estimate Order

### What it is
Get order cost estimate.

### What it does
Get order cost estimate

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Slant3D Estimate Shipping

### What it is
Get shipping cost estimate.

### What it does
Get shipping cost estimate

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Slant3D Get Orders

### What it is
Get all orders for the account.

### What it does
Get all orders for the account

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
<!-- END MANUAL -->

### Outputs
| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| orders | List of orders with their details | List[str] |

### Possible use case
<!-- MANUAL: use_case -->
_Add practical use case examples here._
<!-- END MANUAL -->

---

## Slant3D Tracking

### What it is
Track order status and shipping.

### What it does
Track order status and shipping

### How it works
<!-- MANUAL: how_it_works -->
_Add technical explanation here._
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
_Add practical use case examples here._
<!-- END MANUAL -->

---
