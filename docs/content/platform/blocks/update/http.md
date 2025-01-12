

## Send Web Request

### What it is
A block that allows you to send HTTP requests to web servers and handle their responses.

### What it does
This block sends web requests to specified URLs, supporting various request methods (GET, POST, PUT, etc.), and processes the server's responses. It can handle both JSON and plain text data formats and manages different types of responses including successful responses and error states.

### How it works
1. Takes a URL and request details as input
2. Formats the request according to specified parameters
3. Sends the request to the specified web server
4. Receives the response from the server
5. Processes the response based on its status code
6. Returns the appropriate output based on whether the request was successful or encountered errors

### Inputs
- URL: The web address where the request will be sent (e.g., "https://api.example.com")
- Method: The type of HTTP request to make (GET, POST, PUT, DELETE, PATCH, OPTIONS, or HEAD)
- Headers: Additional information sent with the request, such as authentication tokens or content type specifications
- JSON Format: A yes/no option to specify whether the request and response should be treated as JSON data
- Body: The main content of the request being sent to the server

### Outputs
- Response: The data received from the server when the request is successful (2xx status codes)
- Client Error: Error information received when there's a client-side problem (4xx status codes)
- Server Error: Error information received when there's a server-side problem (5xx status codes)

### Possible use cases
- Integrating with external APIs (e.g., sending customer data to a CRM system)
- Fetching data from web services (e.g., getting weather information)
- Submitting forms to web servers
- Updating information in remote databases
- Validating web services by checking their responses
- Automated testing of web endpoints
- Synchronizing data between different systems

