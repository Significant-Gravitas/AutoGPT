"""Selenium web scraping module."""
from selenium import webdriver
import autogpt.processing.text as summary
from bs4 import BeautifulSoup
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.safari.options import Options as SafariOptions
import logging
from pathlib import Path
from autogpt.config import Config
from typing import List, Tuple, Union

FILE_DIR = Path(__file__).parent.parent
CFG = Config()

browser = None

def get_browser_instance() -> WebDriver:
    """Create or retrieve the browser instance."""
    global browser
    if not browser:
        logging.getLogger("selenium").setLevel(logging.CRITICAL)

        options_available = {'chrome': ChromeOptions, 'safari': SafariOptions, 'firefox': FirefoxOptions}
        options = options_available[CFG.selenium_web_browser]()
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
        )
        options.add_argument("--headless")  # Add headless argument to hide the browser

        if CFG.selenium_web_browser == "firefox":
            browser = webdriver.Firefox(
                executable_path=GeckoDriverManager().install(), options=options
            )
        elif CFG.selenium_web_browser == "safari":
            browser = webdriver.Safari(options=options)
        else:
            browser = webdriver.Chrome(
                executable_path=ChromeDriverManager().install(), options=options
            )

    return browser


def browse_website(url: str, question: str) -> Tuple[str, WebDriver]:
    """Browse a website and return the answer and links to the user

    Args:
        url (str): The url of the website to browse
        question (str): The question asked by the user

    Returns:
        Tuple[str, WebDriver]: The answer and links to the user and the webdriver
    """
    driver, text = scrape_text_with_selenium(url)
    add_header(driver)
    summary_text = summary.summarize_text(url, text, question, driver)
    links = scrape_links_with_selenium(driver)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]

    return f"Answer gathered from website: {summary_text} \n \n Links: {links}", driver


def scrape_text_with_selenium(url: str) -> Tuple[WebDriver, str]:
    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape

    Returns:
        Tuple[WebDriver, str]: The webdriver and the text scraped from the website
    """
    driver = get_browser_instance()
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Get the HTML content directly from the browser's DOM
    page_source = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return driver, text


def scrape_links_with_selenium(driver: WebDriver) -> List[str]:
    """Scrape links from a website using selenium

    Args:
        driver (WebDriver): The webdriver to use to scrape the links

    Returns:
        List[str]: The links scraped from the website
    """
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup)

    return format_hyperlinks(hyperlinks)


def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    driver.quit()


def extract_hyperlinks(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    """Extract hyperlinks from a BeautifulSoup object

    Args:
        soup (BeautifulSoup): The BeautifulSoup object to extract the hyperlinks from

    Returns:
        List[Tuple[str, str]]: The hyperlinks extracted from the BeautifulSoup object
    """
    return [(link.text, link["href"]) for link in soup.find_all("a", href=True)]


def format_hyperlinks(hyperlinks: List[Tuple[str, str]]) -> List[str]:
    """Format hyperlinks to be displayed to the user

    Args:
        hyperlinks (List[Tuple[str, str]]): The hyperlinks to format

    Returns:
        List[str]: The formatted hyperlinks
    """
    return [f"{link_text} ({link_url})" for link_text, link_url in hyperlinks]


def add_header(driver: WebDriver) -> None:
    """Add a header to the website

    Args:
        driver (WebDriver): The webdriver to use to add the header

    Returns:
        None
    """
    driver.execute_script(open(f"{FILE_DIR}/js/overlay.js", "r").read())


def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    global browser
    if browser:
        browser.quit()
        browser = None
