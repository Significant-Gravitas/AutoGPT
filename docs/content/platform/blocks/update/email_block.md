
## Send Web Request

### What it is
A utility block that allows you to send HTTP requests to external web services or APIs.

### What it does
This block enables communication with external web services by sending HTTP requests and handling their responses. It supports various request methods (GET, POST, PUT, etc.) and can handle both JSON and plain text data formats.

### How it works
The block takes your specified parameters (URL, method, headers, etc.), sends a request to the target web service, and processes the response. It automatically handles different types of responses based on the status code received and can work with both JSON and plain text formats.

### Inputs
- URL: The web address where you want to send the request (e.g., "https://api.example.com")
- Method: The type of HTTP request to make (GET, POST, PUT, DELETE, PATCH, OPTIONS, or HEAD)
- Headers: Additional information to send with the request, such as authentication tokens or content type specifications
- JSON Format: A toggle to specify whether the request and response should be treated as JSON data (turned on by default)
- Body: The main content to send with your request (optional)

### Outputs
- Response: The successful response from the server (for status codes 200-299)
- Client Error: Error information when the request fails due to client-side issues (status codes 400-499)
- Server Error: Error information when the request fails due to server-side issues (status codes 500-599)

### Possible use cases
1. Integrating with external APIs:
   - Sending data to a payment processing service
   - Retrieving weather information from a weather API
   - Posting messages to social media platforms
   - Updating inventory in an e-commerce system
   - Fetching customer data from a CRM system

2. Data exchange:
   - Syncing information between different systems
   - Submitting form data to a remote server
   - Downloading content from web services
   - Updating remote databases
   - Sending notifications to external services

### Notes
- The block automatically detects and handles JSON formatting, converting between JSON and plain text as needed
- It provides separate outputs for different types of responses, making it easier to handle both successful and error scenarios
- The block is part of the OUTPUT category, indicating its role in external communications

