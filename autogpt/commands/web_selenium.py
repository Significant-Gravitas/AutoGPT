"""Selenium web scraping module."""
from __future__ import annotations

import logging
from pathlib import Path
from sys import platform
from typing import Optional, Type

from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeDriverService
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeDriverService
from selenium.webdriver.edge.webdriver import WebDriver as EdgeDriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as GeckoDriverService
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.safari.webdriver import WebDriver as SafariDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager as EdgeDriverManager

from autogpt.agents.agent import Agent
from autogpt.command_decorator import command
from autogpt.logs import logger
from autogpt.memory.vector import MemoryItem, get_memory
from autogpt.processing.html import extract_tag_links, format_links
from autogpt.url_utils.validators import validate_url

BrowserOptions = ChromeOptions | EdgeOptions | FirefoxOptions | SafariOptions

FILE_DIR = Path(__file__).parent.parent


@command(
    "browse_website",
    "Browses a Website",
    {
        "url": {"type": "string", "description": "The URL to visit", "required": True},
        "question": {
            "type": "string",
            "description": "What you want to find on the website",
            "required": True,
        },
    },
    aliases=["browse"],
)
@validate_url
def browse_website(url: str, question: str, agent: Agent) -> str:
    """Browse a website and return the answer and links to the user

    Args:
        url (str): The url of the website to browse
        question (str): The question asked by the user

    Returns:
        Tuple[str, WebDriver]: The answer and links to the user and the webdriver
    """
    try:
        driver = get_webdriver(agent)
        text = scrape_text(url, agent, driver)
    except WebDriverException as e:
        # These errors are often quite long and include lots of context.
        # Just grab the first line.
        msg = e.msg.split("\n")[0]
        return f"Error: {msg}"

    add_header(driver)
    summary = summarize_memorize_webpage(url, text, question, agent, driver)
    links = scrape_links(url, agent, driver=driver)

    # Limit links to 5
    if len(links) > 5:
        links = links[:5]
    close_browser(driver)
    return f"Answer gathered from website: {summary}\n\nLinks: {links}"


def get_webdriver(agent: Agent) -> WebDriver:
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options_available: dict[str, Type[BrowserOptions]] = {
        "chrome": ChromeOptions,
        "edge": EdgeOptions,
        "firefox": FirefoxOptions,
        "safari": SafariOptions,
    }

    options: BrowserOptions = options_available[agent.config.selenium_web_browser]()
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36"
    )

    if agent.config.selenium_web_browser == "firefox":
        if agent.config.selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        driver = FirefoxDriver(
            service=GeckoDriverService(GeckoDriverManager().install()), options=options
        )
    elif agent.config.selenium_web_browser == "edge":
        driver = EdgeDriver(
            service=EdgeDriverService(EdgeDriverManager().install()), options=options
        )
    elif agent.config.selenium_web_browser == "safari":
        # Requires a bit more setup on the users end
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari
        driver = SafariDriver(options=options)
    else:
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if agent.config.selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")

        chromium_driver_path = Path("/usr/bin/chromedriver")

        driver = ChromeDriver(
            service=ChromeDriverService(str(chromium_driver_path))
            if chromium_driver_path.exists()
            else ChromeDriverService(ChromeDriverManager().install()),
            options=options,
        )
    return driver


def scrape_text(url: str, agent: Agent, driver: Optional[WebDriver] = None) -> str:
    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape
        agent (Agent): The agent to use
        driver (Optional[WebDriver], optional): The webdriver to use. Defaults to None.
    Returns:
        Tuple[WebDriver, str]: The webdriver and the text scraped from the website
    """
    if driver is None:
        driver = get_webdriver(agent)

    load_url(driver, url)

    # Get the HTML content directly from the browser's DOM
    page_source = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return text


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


def summarize_memorize_webpage(
    url: str,
    text: str,
    question: str,
    agent: Agent,
    driver: Optional[WebDriver] = None,
) -> str:
    """Summarize text using the OpenAI API

    Args:
        url (str): The url of the text
        text (str): The text to summarize
        question (str): The question to ask the model
        driver (WebDriver): The webdriver to use to scroll the page

    Returns:
        str: The summary of the text
    """
    if not text:
        return "Error: No text to summarize"

    text_length = len(text)
    logger.info(f"Text length: {text_length} characters")

    memory = get_memory(agent.config)

    new_memory = MemoryItem.from_webpage(text, url, agent.config, question=question)
    memory.add(new_memory)
    return new_memory.summary


@command(
    "scrape_links",
    "Scrapes a Website for Links",
    {
        "url": {"type": "string", "description": "The URL to visit", "required": True},
        "include_keywords": {
            "type": "list",
            "description": "Keywords, any of which must appear in the link text",
            "required": False,
        },
        "exclude_keywords": {
            "type": "list",
            "description": "Keywords, none of which may appear in the link text",
            "required": False,
        },
    },
    aliases=["extract_links", "extract_urls"],
)
def scrape_links(
    url: str,
    include_keywords: list[str] = [],
    exclude_keywords: list[str] = [],
    driver: Optional[WebDriver] = None,
) -> list[str]:
    """
    Scrape elements from a website using selenium

    Args:
        url (str): The url of the website to scrape
        agent (Agent): The agent to use
        include_keywords (list[str], optional): Keywords, any of which must appear in the element text. Defaults to [].
        exclude_keywords (list[str], optional): Keywords, none of which may appear in the element text. Defaults to [].
        driver (Optional[WebDriver], optional): The webdriver to use. Defaults to None.
    """
    return scrape_tag_links(url, "a", agent, include_keywords, exclude_keywords, driver)


@command(
    "scrape_image_links",
    "Scrapes a Website for Image Tags and Returns the Image Links",
    {
        "url": {"type": "string", "description": "The URL to visit", "required": True},
        "include_keywords": {
            "type": "list",
            "description": "Keywords, any of which must appear in the image srcs",
            "required": False,
        },
        "exclude_keywords": {
            "type": "list",
            "description": "Keywords, none of which may appear in the image srcs",
            "required": False,
        },
    },
    aliases=[
        "extract_images",
        "scrape_images",
        "extract_image_links",
        "extract_image_srcs",
        "scrape_image_sources",
        "scrape_image_srcs",
        "scrape_image_urls",
        "scrape_image_links",
    ],
)
def scrape_image_links(
    url: str,
    include_keywords: list[str] = [],
    exclude_keywords: list[str] = [],
    driver: Optional[WebDriver] = None,
) -> list[str]:
    """
    Scrape image srcs from a website using selenium

    Args:
        url (str): The url of the website to scrape
        agent (Agent): The agent to use
        include_keywords (list[str], optional): Keywords, any of which must appear in the element text. Defaults to [].
        exclude_keywords (list[str], optional): Keywords, none of which may appear in the element text. Defaults to [].
        driver (Optional[WebDriver], optional): The webdriver to use. Defaults to None.
    """
    return scrape_tag_links(
        url, "img", agent, include_keywords, exclude_keywords, driver
    )


@validate_url
def scrape_tag_links(
    url: str,
    tag_type: str,
    agent: Agent,
    include_keywords: list[str] = [],
    exclude_keywords: list[str] = [],
    driver: Optional[WebDriver] = None,
) -> list[str]:
    """
    Scrape elements from a website using selenium

    Args:
        url (str): The url of the website to scrape
        agent (Agent): The agent to use
        include_keywords (list[str], optional): Keywords, any of which must appear in the element text. Defaults to [].
        exclude_keywords (list[str], optional): Keywords, none of which may appear in the element text. Defaults to [].
        driver (Optional[WebDriver], optional): The webdriver to use. Defaults to None.
    """
    if driver is None:
        driver = get_webdriver(agent)

    load_url(driver, url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    page_source = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    elements = extract_tag_links(
        soup, url, tag_type, include_keywords, exclude_keywords
    )
    return format_links(elements)


def load_url(driver, url):
    if driver.current_url != url:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
