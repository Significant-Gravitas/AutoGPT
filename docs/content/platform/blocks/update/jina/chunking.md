
<file_name>autogpt_platform/backend/backend/blocks/jina/search.md</file_name>

## Search The Web

### What it is
A specialized block that performs web searches using the Jina.ai search engine.

### What it does
This block takes a search query and returns relevant search results from across the internet, including content from the top 5 URLs.

### How it works
The block connects to the Jina.ai search service, sends your search query securely using provided credentials, and retrieves organized search results from various web sources.

### Inputs
- Credentials: Authentication details required to access the Jina.ai search service
- Query: The search term or phrase you want to look up on the internet

### Outputs
- Results: A comprehensive collection of search findings, including content from the top 5 relevant URLs
- Error: Information about what went wrong if the search fails to complete

### Possible use case
A researcher looking to gather information about recent developments in renewable energy could use this block to quickly collect relevant content from multiple authoritative sources.

## Extract Website Content

### What it is
A block designed to collect and extract content from specific web pages.

### What it does
This block retrieves and processes content from a given URL, either using raw scraping or the more sophisticated Jina-ai Reader method.

### How it works
The block accepts a URL and connects to the webpage, then either performs a direct content extraction or uses Jina-ai's specialized reader to gather and organize the content in a more structured way.

### Inputs
- Credentials: Authentication details required to access the Jina.ai service
- URL: The web address of the page you want to extract content from
- Raw Content: An optional setting that determines whether to use basic scraping (true) or enhanced Jina-ai Reader extraction (false)

### Outputs
- Content: The extracted text and information from the specified webpage
- Error: Information about what went wrong if the content extraction fails

### Possible use case
A content curator building a knowledge base could use this block to automatically extract and save relevant information from multiple web articles about a specific topic, saving time and ensuring consistency in content collection.

