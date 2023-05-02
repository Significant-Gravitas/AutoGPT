"""Selenium web scraping module."""
from __future__ import annotations

import logging
from pathlib import Path
from sys import platform

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

import autogpt.processing.text as summary
from autogpt.commands.command import command
from autogpt.config import Config
from autogpt.processing.html import extract_hyperlinks, format_hyperlinks
from autogpt.url_utils.validators import validate_url

FILE_DIR = Path(__file__).parent.parent
CFG = Config()


@command(
    "browse_website",
    "Browse Website",
    '"url": "<url>", "question": "<what_you_want_to_find_on_website>"',
)
@validate_url
def browse_website(url: str, question: str) -> tuple[str, WebDriver]:
    """Browse a website and return the answer and links to the user

    Args:
        url (str): The url of the website to browse
        question (str): The question asked by the user

    Returns:
        Tuple[str, WebDriver]: The answer and links to the user and the webdriver
    """
    try:
        driver, text = scrape_text_with_selenium(url)
    except WebDriverException as e:
        # These errors are often quite long and include lots of context.
        # Just grab the first line.
        msg = e.msg.split("\n")[0]
        return f"Error: {msg}", None

    add_header(driver)
    summary_text = summary.summarize_text(url, text, question, driver)
    links = scrape_links_with_selenium(driver, url)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]
    close_browser(driver)
    return f"Answer gathered from website: {summary_text} \n \n Links: {links}", driver


def scrape_text_with_selenium(url: str) -> tuple[WebDriver, str]:
    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape

    Returns:
        Tuple[WebDriver, str]: The webdriver and the text scraped from the website
    """
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options_available = {
        "chrome": ChromeOptions,
        "safari": SafariOptions,
        "firefox": FirefoxOptions,
    }

    options = options_available[CFG.selenium_web_browser]()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    if CFG.selenium_web_browser == "firefox":
        if CFG.selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        driver = webdriver.Firefox(
            executable_path=GeckoDriverManager().install(), options=options
        )
    elif CFG.selenium_web_browser == "safari":
        # Requires a bit more setup on the users end
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = webdriver.Safari(options=options)
    else:
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if CFG.selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")

        chromium_driver_path = Path("/usr/bin/chromedriver")

        driver = webdriver.Chrome(
            executable_path=chromium_driver_path
            if chromium_driver_path.exists()
            else ChromeDriverManager().install(),
            options=options,
        )
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


def scrape_links_with_selenium(driver: WebDriver, url: str) -> list[str]:
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

    hyperlinks = extract_hyperlinks(soup, url)

    return format_hyperlinks(hyperlinks)


def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    driver.quit()


def add_header(driver: WebDriver) -> None:
    """Add a header to the website

    Args:
        driver (WebDriver): The webdriver to use to add the header

    Returns:
        None
    """
    try:
        with open(f"{FILE_DIR}/js/overlay.js", "r") as overlay_file:
            overlay_script = overlay_file.read()
        driver.execute_script(overlay_script)
    except Exception as e:
        print(f"Error executing overlay.js: {e}")
