# HTML processing functions

This module contains two functions for processing HTML: `extract_hyperlinks` and `format_hyperlinks`.

## `extract_hyperlinks`

```python
def extract_hyperlinks(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    """Extract hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object
        base_url (str): The base URL

    Returns:
        List[Tuple[str, str]]: The extracted hyperlinks
    """
```

This function extracts hyperlinks from a [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#beautifulsoup) object and returns them as a list of tuples. The first element of each tuple is the text of the link, and the second element is the absolute URL of the link.

### Arguments

* `soup` (required): A [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#beautifulsoup) object representing the HTML to extract hyperlinks from.
* `base_url` (required): A string representing the base URL for the hyperlinks. This is used to resolve relative URL paths.

### Returns

* A list of tuples, where each tuple contains the text of the link and the absolute URL of the link.

### Example

```python
from bs4 import BeautifulSoup
from requests.compat import urljoin
from html_processing import extract_hyperlinks

# Load HTML into BeautifulSoup object
html = '<html><body><a href="/about">About us</a><a href="/contact">Contact us</a></body></html>'
soup = BeautifulSoup(html, 'html.parser')

# Extract hyperlinks
hyperlinks = extract_hyperlinks(soup, 'https://www.example.com')

# Print hyperlinks
for link_text, link_url in hyperlinks:
    print(f'{link_text}: {link_url}')
```

Output:
```
About us: https://www.example.com/about
Contact us: https://www.example.com/contact
```

## `format_hyperlinks`

```python
def format_hyperlinks(hyperlinks: list[tuple[str, str]]) -> list[str]:
    """Format hyperlinks to be displayed to the user

    Args:
        hyperlinks (List[Tuple[str, str]]): The hyperlinks to format

    Returns:
        List[str]: The formatted hyperlinks
    """
```

This function formats a list of hyperlinks (as returned by `extract_hyperlinks`) into a list of strings that can be displayed to the user. The format of each string is `link_text (link_url)`.

### Arguments

* `hyperlinks` (required): A list of tuples representing the hyperlinks, where the first element of each tuple is the link text and the second element is the link URL.

### Returns

* A list of strings, where each string represents a formatted hyperlink.

### Example

```python
from html_processing import format_hyperlinks

# Hyperlinks returned by extract_hyperlinks
hyperlinks = [('About us', 'https://www.example.com/about'), ('Contact us', 'https://www.example.com/contact')]

# Format hyperlinks for display
formatted_hyperlinks = format_hyperlinks(hyperlinks)

# Print formatted hyperlinks
for link in formatted_hyperlinks:
    print(link)
```

Output:
```
About us (https://www.example.com/about)
Contact us (https://www.example.com/contact)
```