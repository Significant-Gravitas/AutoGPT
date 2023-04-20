## Browse a webpage and summarize it using the LLM model

This module provides functionality to browse a webpage and summarize it using the LLM model.

### Functions

#### `is_valid_url(url: str) -> bool`

Check if the URL is valid.

##### Arguments

- `url` (`str`): The URL to check.

##### Returns

- `bool`: `True` if the URL is valid, `False` otherwise.

#### `sanitize_url(url: str) -> str`

Sanitize the URL.

##### Arguments

- `url` (`str`): The URL to sanitize.

##### Returns

- `str`: The sanitized URL.

#### `check_local_file_access(url: str) -> bool`

Check if the URL is a local file.

##### Arguments

- `url` (`str`): The URL to check.

##### Returns

- `bool`: `True` if the URL is a local file, `False` otherwise.

#### `get_response(url: str, timeout: int = 10) -> tuple[None, str] | tuple[Response, None]`

Get the response from a URL.

##### Arguments

- `url` (`str`): The URL to get the response from.
- `timeout` (`int`, optional, default=10): The timeout for the HTTP request.

##### Returns

- `tuple[None, str] | tuple[Response, None]`: The response and error message.

##### Raises

- `ValueError`: If the URL is invalid.
  `requests.exceptions.RequestException`: If the HTTP request fails.

#### `scrape_text(url: str) -> str`

Scrape text from a webpage.

##### Arguments

- `url` (`str`): The URL to scrape text from.

##### Returns

- `str`: The scraped text.

#### `scrape_links(url: str) -> str | list[str]`

Scrape links from a webpage.

##### Arguments

- `url` (`str`): The URL to scrape links from.

##### Returns

- `str | list[str]`: The scraped links.

#### `create_message(chunk, question)`

Create a message for the user to summarize a chunk of text.

##### Arguments

- `chunk` (`str`): The text to summarize.
- `question` (`str`): The question to answer.

##### Returns

- `dict`: The message as a dictionary. 

### Example

```python
from autogpt.llm_scraping import is_valid_url, sanitize_url, get_response, scrape_text, scrape_links, create_message

# example url
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

# check if url is valid
print(is_valid_url(url)) # True

# sanitize url
print(sanitize_url(url)) # https://en.wikipedia.org/wiki/Python_(programming_language)

# get response from url and scrape text
response, error_message = get_response(url)
if error_message:
    print(error_message)
if not response:
    print("Error: Could not get response")
text = scrape_text(url)
print(text)

# scrape links from url
links = scrape_links(url)
print(links)

# create message for user to summarize text
message = create_message(text, "What are the features of Python?")
print(message)
```