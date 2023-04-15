import logging
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from autogpt.config import Config
from autogpt.llm_utils import create_chat_completion
from autogpt.memory import get_memory

cfg = Config()
memory = get_memory(cfg)
_browser_instance = None

# Function to check if the URL is valid
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


# Function to sanitize the URL
def sanitize_url(url):
    return urljoin(url, urlparse(url).path)


# Define and check for local file address prefixes
def check_local_file_access(url):
    local_prefixes = [
        "file:///",
        "file://localhost",
        "http://localhost",
        "https://localhost",
    ]
    return any(url.startswith(prefix) for prefix in local_prefixes)


def check_and_sanitize_url(url):
    # Restrict access to local files
    if check_local_file_access(url):
        raise ValueError("Access to local files is restricted")

    # Most basic check if the URL is valid:
    if not url.startswith("http://") and not url.startswith("https://"):
        raise ValueError("Invalid URL format")

    sanitized_url = sanitize_url(url)
    return sanitized_url


def scrape_text(url):
    """Scrape text from a webpage"""
    browser = get_browser_instance()
    try:
        page_source = browser.get_page_source(url)
    except ValueError as ve:
        return str(ve)

    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


def extract_hyperlinks(soup):
    """Extract hyperlinks from a BeautifulSoup object"""
    hyperlinks = []
    for link in soup.find_all("a", href=True):
        hyperlinks.append((link.text, link["href"]))
    return hyperlinks


def format_hyperlinks(hyperlinks):
    """Format hyperlinks into a list of strings"""
    formatted_links = []
    for link_text, link_url in hyperlinks:
        formatted_links.append(f"{link_text} ({link_url})")
    return formatted_links


def scrape_links(url):
    """Scrape links from a webpage"""
    browser = get_browser_instance()
    try:
        page_source = browser.get_page_source(url)
    except ValueError as ve:
        return str(ve)

    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)


def split_text(text, max_length=cfg.browse_chunk_max_length):
    """Split text into chunks of a maximum length"""
    paragraphs = text.split("\n")
    current_length = 0
    current_chunk = []

    for paragraph in paragraphs:
        if current_length + len(paragraph) + 1 <= max_length:
            current_chunk.append(paragraph)
            current_length += len(paragraph) + 1
        else:
            yield "\n".join(current_chunk)
            current_chunk = [paragraph]
            current_length = len(paragraph) + 1

    if current_chunk:
        yield "\n".join(current_chunk)


def create_message(chunk, question):
    """Create a message for the user to summarize a chunk of text"""
    return {
        "role": "user",
        "content": f'"""{chunk}""" Using the above text, please answer the following'
        f' question: "{question}" -- if the question cannot be answered using the'
        " text, please summarize the text.",
    }


def summarize_text(url, text, question):
    """Summarize text using the LLM model"""
    if not text:
        return "Error: No text to summarize"

    text_length = len(text)
    print(f"Text length: {text_length} characters")

    summaries = []
    chunks = list(split_text(text))

    for i, chunk in enumerate(chunks):
        print(f"Adding chunk {i + 1} / {len(chunks)} to memory")

        memory_to_add = f"Source: {url}\n" f"Raw content part#{i + 1}: {chunk}"

        memory.add(memory_to_add)

        print(f"Summarizing chunk {i + 1} / {len(chunks)}")
        messages = [create_message(chunk, question)]

        summary = create_chat_completion(
            model=cfg.fast_llm_model,
            messages=messages,
            max_tokens=cfg.browse_summary_max_token,
        )
        summaries.append(summary)
        print(f"Added chunk {i + 1} summary to memory")

        memory_to_add = f"Source: {url}\n" f"Content summary part#{i + 1}: {summary}"

        memory.add(memory_to_add)

    print(f"Summarized {len(chunks)} chunks.")

    combined_summary = "\n".join(summaries)
    messages = [create_message(combined_summary, question)]

    final_summary = create_chat_completion(
        model=cfg.fast_llm_model,
        messages=messages,
        max_tokens=cfg.browse_summary_max_token,
    )

    return final_summary


def _initialize_requests_session():
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent})
    return session


def get_browser_instance():
    # Singleton derived from "BrowserBase" class
    global _browser_instance

    if _browser_instance is not None:
        return _browser_instance
    else:
        browser_cls = {'HeadlessBarebones': BrowserBareBonesHeadless,
                       'SeleniumChrome': BrowserSeleniumChrome,
                       }
        assert cfg.browser_automation in list(browser_cls.keys()), \
            'ERROR: Unknown browser setting for BROWSER_AUTOMATION in .env config file.'
        cls = browser_cls[cfg.browser_automation]
        _browser_instance = cls()
        return _browser_instance


class BrowserBase(ABC):

    @abstractmethod
    def get_page_source(self, url):
        # Return the body of the HTML page, in a format that can be digested by BeautifulSoup's html.parser
        # If an error was encountered, then raise an ValueError exception for simplicity.
        pass


class BrowserBareBonesHeadless(BrowserBase):
    """
    Advantage: headless, so runs on servers too. Does not need Chrome installed. Faster
    Disadvantage: May miss some content, because the Javascript parts of the website are not executed.
    """
    session = _initialize_requests_session()

    def __init__(self):
        pass

    def get_page_source(self, url):
        """Scrape text from a webpage"""
        response, error_message = self.get_response(url)
        if error_message:
            raise ValueError(error_message)

        if not response:
            raise ValueError("Error: Could not get response")

        return response.text

    def get_response(self, url, timeout=10):
        try:

            sanitized_url = check_and_sanitize_url(url)
            response = BrowserBareBonesHeadless.session.get(sanitized_url, timeout=timeout)

            # Check if the response contains an HTTP error
            if response.status_code >= 400:
                return None, "Error: HTTP " + str(response.status_code) + " error"

            return response, None
        except ValueError as ve:
            # Handle invalid URL format
            return None, "Error: " + str(ve)

        except requests.exceptions.RequestException as re:
            # Handle exceptions related to the HTTP request
            #  (e.g., connection errors, timeouts, etc.)
            return None, "Error: " + str(re)


class BrowserSeleniumChrome(BrowserBase):
    """
    Advantage: will load a website with Javascript running, as many modern websites need this for proper content
    Disadvantage: can be slower than the BrowserBareBonesHeadless
    """
    file_dir = Path(__file__).parent

    def __init__(self):
        logging.getLogger("selenium").setLevel(logging.CRITICAL)
        self.options = Options()
        self.options.add_argument(f"user-agent={cfg.user_agent}")

    def get_page_source(self, url):
        # TODO: re-use a session in Selenium, instead of starting a new one every time
        sanitized_url = check_and_sanitize_url(url)
        with webdriver.Chrome(executable_path=ChromeDriverManager().install(),
                              options=self.options) as driver:
            driver.get(sanitized_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            # Get the HTML content directly from the browser's DOM
            page_source = driver.execute_script("return document.body.outerHTML;")

            # Add graphical overlay
            self.add_header(driver)

            # Close browser
            driver.quit()

        return page_source

    def add_header(self, driver):
        driver.execute_script(open(f"{self.file_dir}/js/overlay.js", "r").read())
