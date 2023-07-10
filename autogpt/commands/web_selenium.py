"""Selenium web scraping module."""
from __future__ import annotations

import logging
from pathlib import Path
from sys import platform
from typing import Optional, Type, Callable

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

from autogpt.agent.agent import Agent
from autogpt.command_decorator import command
from autogpt.logs import logger
from autogpt.memory.vector import MemoryItem, get_memory
from autogpt.processing.html import extract_hyperlinks, extract_image_links, extract_tag_links, format_links
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
        webdriver = get_webdriver(agent)
        webdriver, text = scrape_text_with_selenium(url, agent, webdriver)
    except WebDriverException as e:
        # These errors are often quite long and include lots of context.
        # Just grab the first line.
        msg = e.msg.split("\n")[0]
        return f"Error: {msg}"

    add_header(webdriver)
    summary = summarize_memorize_webpage(url, text, question, agent, webdriver)
    links = scrape_links_with_selenium(url, agent, webdriver=webdriver)
    
    # Limit links to 5
    if len(links) > 5:
        links = links[:5]
    close_browser(webdriver)
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

def scrape_text_with_selenium(url: str, agent: Agent, webdriver: Optional[WebDriver]=None) -> tuple[WebDriver, str]:
    """Scrape text from a website using selenium

    Args:
        url (str): The url of the website to scrape
        agent (Agent): The agent to use
        webdriver (Optional[WebDriver], optional): The webdriver to use. Defaults to None.
    Returns:
        Tuple[WebDriver, str]: The webdriver and the text scraped from the website
    """
    if webdriver is None:
        webdriver = get_webdriver(agent)
        
    webdriver.get(url)

    WebDriverWait(webdriver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # Get the HTML content directly from the browser's DOM
    page_source = webdriver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)
    return webdriver, text

@command(
    "scrape_tags",
    "Scrapes a Website for Image or Link Tags",
    {
        "url": {"type": "string", "description": "The URL to visit", "required": True},
        "tag_type": {"type": "string", "description": "The tag to scrape (img or a)", "required": True},
        "include_keywords": {"type": "list", "description": "Keywords, any of which must appear in the image srcs", "required": False},
        "exclude_keywords": {"type": "list", "description": "Keywords, none of which may appear in the image srcs", "required": False},
    },
)
@validate_url
def scrape_tags_with_selenium(url: str, tag_type: str, agent: Agent, include_keywords: list[str]=[], exclude_keywords: list[str]=[], webdriver: Optional[WebDriver]=None) -> list[str]:
    """
    Scrape elements from a website using selenium
    
    Args:
        url (str): The url of the website to scrape
        agent (Agent): The agent to use
        extraction_func (Callable): The function to use to extract elements from the website
        formatting_func (Callable): The function to use to format the elements
        include_keywords (list[str], optional): Keywords, any of which must appear in the element text. Defaults to [].
        exclude_keywords (list[str], optional): Keywords, none of which may appear in the element text. Defaults to [].
        webdriver (Optional[WebDriver], optional): The webdriver to use. Defaults to None.
    """
    if webdriver is None:
        webdriver = get_webdriver(agent)
        
    webdriver.get(url)
    
    WebDriverWait(webdriver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    page_source = webdriver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()
        
    elements = extract_tag_links(soup, url, tag_type, include_keywords, exclude_keywords)
    return format_links(elements)

def scrape_images_with_selenium(url: str, agent: Agent, include_keywords: list[str]=[], exclude_keywords: list[str]=[], webdriver: Optional[WebDriver]=None) -> list[str]:
    """Scrape images from a website using selenium
    
    Args:
        url (str): The url of the website to scrape
        agent (Agent): The agent to use
        include_keywords (list[str], optional): Keywords, any of which must appear in the links. Defaults to [].
        exclude_keywords (list[str], optional): Keywords, none of which may appear in the links. Defaults to [].
        webdriver (WebDriver): The webdriver to use to scrape the links
    
    Returns:
        List[str]: The image links scraped from the website
    """
    return scrape_tags_with_selenium(url, "img", agent, extract_image_links, format_links, include_keywords, exclude_keywords, webdriver)

def scrape_links_with_selenium(url: str, agent: Agent, include_keywords: list[str]=[], exclude_keywords: list[str]=[], webdriver: Optional[WebDriver]=None) -> list[str]:
    """Scrape links from a website using selenium

    Args:
        url (str): The url of the website to scrape
        agent (Agent): The agent to use
        include_keywords (list[str], optional): Keywords, any of which must appear in the links. Defaults to [].
        exclude_keywords (list[str], optional): Keywords, none of which may appear in the links. Defaults to [].
        webdriver (WebDriver): The webdriver to use to scrape the links

    Returns:
        List[str]: The links scraped from the website
    """
    return scrape_tags_with_selenium(url, "a", agent, extract_hyperlinks, format_links, include_keywords, exclude_keywords, webdriver)

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
