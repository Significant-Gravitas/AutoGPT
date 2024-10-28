# Send Web Request

## What it is
The Send Web Request block is a tool for making HTTP requests to specified web addresses.

## What it does
This block allows you to send various types of web requests (such as GET, POST, PUT, etc.) to a given URL, optionally including headers and a request body. It then processes the response and categorizes it based on the status code received.

## How it works
When activated, the block takes the provided URL, request method, headers, and body. It then sends the request to the specified web address. Upon receiving a response, it analyzes the status code and returns the response data in one of three categories: successful response, client error, or server error.

## Inputs
| Input | Description |
|-------|-------------|
| URL | The web address to which the request will be sent |
| Method | The type of HTTP request (e.g., GET, POST, PUT). Default is POST |
| Headers | Additional information sent with the request, such as authentication tokens or content type. This is optional |
| Body | The main content of the request, typically used for sending data in POST or PUT requests. This is optional |

## Outputs
| Output | Description |
|--------|-------------|
| Response | The data received from a successful request (status codes 200-299) |
| Client Error | Information about errors caused by the client, such as invalid requests (status codes 400-499) |
| Server Error | Information about errors on the server side (status codes 500-599) |

## Possible use case
This block could be used in an application that needs to interact with external APIs. For example, it could send user data to a registration service, retrieve product information from an e-commerce platform, or post updates to a social media service. The block's ability to handle different types of responses makes it versatile for various web-based interactions.