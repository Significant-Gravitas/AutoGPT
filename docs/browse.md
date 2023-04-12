## Module `web_scraper.py`

The `web_scraper` module exposes functions to extract textual and hyperlink information from a webpage and summarize the textual information using a language model.

### Dependencies

The following third-party libraries are required:

- `requests`
- `bs4` (BeautifulSoup)

Additionally, the module imports the following custom modules and classes:

- `Config` from `config.py`
- `create_chat_completion` from `llm_utils.py`

### Functions

The module provides the following functions:

#### `scrape_text(url: str) -> str`

Scrapes the textual content from the webpage at the specified URL using `requests` and `BeautifulSoup`.

Params:
- url (str): The URL of the webpage to scrape.

Returns:
- A string containing the textual content of the webpage.

Raises:
- An error message in string format if:
  - URL is invalid
  - Access to local files is restricted
  - HTTP error occurs while accessing the webpage

#### `scrape_links(url: str) -> List[str]`

Scrapes the hyperlinks from the webpage at the specified URL using `requests` and `BeautifulSoup`.

Params:
- url (str): The URL of the webpage to scrape.

Returns:
- A list of strings containing the hyperlinks in the format `link_text (link_url)`.

Raises:
- An error message in string format if an HTTP error occurs while accessing the webpage.

#### `summarize_text(text: str, question: str) -> str`

Summarizes the textual content using a pre-trained language model.

Params:
- text (str): The textual content to summarize.
- question (str): The question to answer based on the text.

Returns:
- A summary of the textual content in string format.

Raises:
- An error message in string format if there is no text to summarize.

#### `split_text(text: str, max_length: int = 8192) -> Iterable[str]`

Splits the input text into smaller chunks of a specified maximum length.

Params:
- text (str): The input text to split.
- max_length (int): The maximum length of each text chunk. Defaults to 8192.

Yields:
- An iterable of strings containing the input text split into chunks of the maximum length.

#### `create_message(chunk: str, question: str) -> Dict[str, str]`

Creates a message for the user to summarize a chunk of text.

Params:
- chunk (str): The chunk of textual content to summarize.
- question (str): The question to answer based on the text.

Returns:
- A dictionary containing the message `role` (fixed to `"user"`) and `content`.

### Example Usage

```python
from web_scraper import scrape_text, summarize_text

url = "https://en.wikipedia.org/wiki/Web_scraping"

# Scrape webpage content
text = scrape_text(url)

# Summarize webpage content
summary = summarize_text(text, "What is web scraping?")
```
