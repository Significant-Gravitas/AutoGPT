
<file_name>autogpt_platform/backend/backend/blocks/helpers/http.md</file_name>

## HTTP Get Request Block

### What it is
A utility block that handles making HTTP GET requests to retrieve data from web services or APIs.

### What it does
This block sends GET requests to specified URLs and returns the response data either as JSON or plain text format.

### How it works
The block takes a URL and optional parameters, connects to that URL using the HTTP GET method, and retrieves the requested data. It can process the response in two ways:
1. Convert it to JSON format for structured data
2. Return it as plain text for unstructured data

### Inputs
- URL: The web address where you want to retrieve data from
- Headers (optional): Additional information sent with the request, such as authentication tokens or content preferences
- JSON Flag (optional): Determines whether to return the response as JSON (true) or plain text (false)

### Outputs
- Response Data: The retrieved information in either JSON format or plain text, depending on the JSON flag setting

### Possible use cases
- Fetching weather data from a weather service API
- Retrieving product information from an e-commerce platform
- Getting social media posts from a social network's API
- Downloading content from web services
- Checking the status of an online service

