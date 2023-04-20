# Web Scraping Commands using Playwright

This module implements two web scraping functions using [Playwright](https://playwright.dev/), `scrape_text` and `scrape_links`, to extract text and links respectively from a given webpage. 

## Installation

Install [Playwright](https://playwright.dev/) and [BeautifulSoup4](https://pypi.org/project/beautifulsoup4/) using `pip`:
```
pip install playwright beautifulsoup4
```

## Usage

### scrape_text

```python
def scrape_text(url: str) -> str:
    """Scrape text from a webpage

    Args:
        url (str): The URL to scrape text from

    Returns:
        str: The scraped text
    """
```

The `scrape_text` function accepts a URL and returns the scraped text from the webpage in the form of a string.

Example usage:
```python
text = scrape_text('https://en.wikipedia.org/wiki/Web_scraping')
print(text)
```

### scrape_links

```python
def scrape_links(url: str) -> str | list[str]:
    """Scrape links from a webpage

    Args:
        url (str): The URL to scrape links from

    Returns:
        Union[str, List[str]]: The scraped links
    """
```

The `scrape_links` function accepts a URL and returns the scraped links from the webpage in the form of a list of strings.

Example usage:
```python
links = scrape_links('https://en.wikipedia.org/wiki/Web_scraping')
print(links)
```