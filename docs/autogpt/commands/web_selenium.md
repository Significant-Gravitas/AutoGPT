## Selenium Web Scraping Module

This module provides functionality for scraping websites using Selenium. Selenium is a tool for automating web browsers, which can be used for web scraping. This module uses Selenium to extract text and links from a website.

### Functionality

#### `browse_website(url: str, question: str) -> Tuple[str, WebDriver]`

This function browses a website and returns the answer and links to the user. It takes in the following parameters: 

- `url` (str): The url of the website to browse 
- `question` (str): The question asked by the user  

It returns: 

- A tuple containing the answer and links to the user and the webdriver.

#### `scrape_text_with_selenium(url: str) -> Tuple[WebDriver, str]`

This function scrapes text from a website using Selenium. It takes in the following parameters: 

- `url` (str): The url of the website to scrape  

It returns: 

- A tuple containing the webdriver and the text scraped from the website.

#### `scrape_links_with_selenium(driver: WebDriver, url: str) -> List[str]`

This function scrapes links from a website using Selenium. It takes in the following parameters: 

- `driver` (WebDriver): The webdriver to use to scrape the links  

It returns: 

- A list of links scraped from the website. 

#### `close_browser(driver: WebDriver) -> None`

This function closes the browser. It takes in the following parameters:

- `driver` (WebDriver): The webdriver to close  

It returns 

- None.

#### `add_header(driver: WebDriver) -> None`

This function adds a header to the website. It takes in the following parameters: 

- `driver` (WebDriver): The webdriver to use to add the header  

It returns: 

- None.

### Example Usage

```python
from selenium_scraping_module import browse_website, scrape_text_with_selenium, scrape_links_with_selenium, close_browser, add_header

url = "http://www.example.com"
question = "What is the website about?"

# Example usage of browse_website function
answer, driver = browse_website(url, question)
print(answer)

# Example usage of scrape_text_with_selenium function
driver, text = scrape_text_with_selenium(url)
print(text)

# Example usage of scrape_links_with_selenium function
links = scrape_links_with_selenium(driver, url)
print(links)

# Example usage of close_browser function
close_browser(driver)

# Example usage of add_header function
add_header(driver)
```