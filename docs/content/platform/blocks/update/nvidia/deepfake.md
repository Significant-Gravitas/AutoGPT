
<file_name>autogpt_platform/backend/backend/blocks/slant3d/order.md</file_name>

# Slant3D Order Management Blocks Documentation

## Slant3D Create Order

### What it is
A block for creating new 3D printing orders through the Slant3D service.

### What it does
Creates a new print order with specified customer details, items, and shipping information.

### How it works
Takes order details including customer information and items to be printed, processes them, and submits them to the Slant3D service to create a new order.

### Inputs
- Credentials: Your Slant3D API authentication credentials
- Order Number: Optional custom order number (automatically generated if not provided)
- Customer Details: Shipping information including name, email, phone, and address
- Items: List of items to be printed, including file URL, quantity, color, and material profile

### Outputs
- Order ID: The unique Slant3D order identifier
- Error: Any error message if the order creation fails

### Possible use case
An e-commerce platform automatically submitting customer 3D printing orders to Slant3D for fulfillment.

## Slant3D Estimate Order

### What it is
A block for calculating cost estimates for 3D printing orders.

### What it does
Provides a detailed cost breakdown including total price, shipping, and printing costs.

### How it works
Submits order details to Slant3D's estimation service and returns a comprehensive cost breakdown.

### Inputs
- Credentials: Your Slant3D API authentication credentials
- Order Number: Optional custom order number
- Customer Details: Shipping information for accurate cost calculation
- Items: List of items to be printed

### Outputs
- Total Price: Complete order cost in USD
- Shipping Cost: Estimated shipping expenses
- Printing Cost: Cost of printing the items
- Error: Any error message if the estimation fails

### Possible use case
Providing instant price quotes to customers before they commit to a 3D printing order.

## Slant3D Estimate Shipping

### What it is
A block specifically for calculating shipping costs.

### What it does
Provides shipping cost estimates based on delivery location and order details.

### How it works
Analyzes the shipping address and order specifications to calculate accurate shipping costs.

### Inputs
- Credentials: Your Slant3D API authentication credentials
- Order Number: Optional custom order number
- Customer Details: Shipping address information
- Items: List of items to determine shipping costs

### Outputs
- Shipping Cost: Estimated shipping expense
- Currency Code: The currency of the quoted price (e.g., 'usd')
- Error: Any error message if the estimation fails

### Possible use case
Calculating shipping costs during checkout for different delivery locations.

## Slant3D Get Orders

### What it is
A block for retrieving all orders associated with an account.

### What it does
Fetches a complete list of orders and their details from the Slant3D system.

### How it works
Retrieves all order information using the provided credentials and returns a comprehensive list.

### Inputs
- Credentials: Your Slant3D API authentication credentials

### Outputs
- Orders: List of all orders with their details
- Error: Any error message if the request fails

### Possible use case
Displaying order history or generating reports for business analytics.

## Slant3D Tracking

### What it is
A block for monitoring order status and tracking shipments.

### What it does
Provides current order status and shipping tracking information.

### How it works
Queries the Slant3D system for real-time order status and tracking details.

### Inputs
- Credentials: Your Slant3D API authentication credentials
- Order ID: The Slant3D order identifier to track

### Outputs
- Status: Current order status
- Tracking Numbers: List of shipping tracking numbers
- Error: Any error message if tracking fails

### Possible use case
Providing customers with real-time updates on their order status and shipping progress.

## Slant3D Cancel Order

### What it is
A block for canceling existing orders.

### What it does
Processes order cancellation requests and confirms the cancellation status.

### How it works
Submits a cancellation request to the Slant3D system for the specified order.

### Inputs
- Credentials: Your Slant3D API authentication credentials
- Order ID: The Slant3D order identifier to cancel

### Outputs
- Status: Cancellation confirmation message
- Error: Any error message if cancellation fails

### Possible use case
Allowing customers to cancel their orders before production begins or handling order modifications.

