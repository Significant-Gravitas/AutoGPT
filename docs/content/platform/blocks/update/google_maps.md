
<file_name>autogpt_platform/backend/backend/blocks/http.md</file_name>

## Send Web Request

### What it is
A versatile block designed to make HTTP requests to any specified URL, allowing for communication with external web services and APIs.

### What it does
This block sends HTTP requests to a specified URL with customizable parameters such as method type, headers, and body content. It can handle both JSON and non-JSON data formats and processes different types of responses based on the status codes received.

### How it works
1. Takes the provided URL and request parameters
2. Determines if the body content should be treated as JSON or plain text
3. Sends the HTTP request to the specified URL
4. Processes the response based on the status code
5. Returns the appropriate response or error message

### Inputs
- URL: The web address where the request will be sent (e.g., "https://api.example.com")
- Method: The type of HTTP request to make (GET, POST, PUT, DELETE, PATCH, OPTIONS, or HEAD), defaults to POST
- Headers: Optional key-value pairs of HTTP headers to include with the request
- JSON Format: A yes/no option to specify whether the request and response should be treated as JSON (defaults to yes)
- Body: The content to be sent with the request (optional)

### Outputs
- Response: The successful response from the server (for 2xx status codes)
- Client Error: Error information when the request fails due to client-side issues (4xx status codes)
- Server Error: Error information when the request fails due to server-side issues (5xx status codes)

### Possible use cases
- Integrating with external APIs (like weather services, payment processors, or social media platforms)
- Sending data to a web service for processing
- Retrieving information from a REST API
- Updating records in a remote database through its API
- Validating web services availability and responses
- Automating interactions with web-based services

