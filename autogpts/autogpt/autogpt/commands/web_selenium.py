"""Commands for browsing a website"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from sys import platform
from typing import TYPE_CHECKING, Optional, Type

from bs4 import BeautifulSoup
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeDriverService
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.options import ArgOptions as BrowserOptions
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

from autogpt.agents.utils.exceptions import CommandExecutionError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.processing.html import extract_hyperlinks, format_hyperlinks
from autogpt.processing.text import summarize_text
from autogpt.url_utils.validators import validate_url

COMMAND_CATEGORY = "web_browse"
COMMAND_CATEGORY_TITLE = "Web Browsing"


if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    from autogpt.config import Config


logger = logging.getLogger(__name__)

FILE_DIR = Path(__file__).parent.parent
TOKENS_TO_TRIGGER_SUMMARY = 50
LINKS_TO_RETURN = 20


class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""


@command(
    "read_webpage",
    (
        "Read a webpage, and extract specific information from it"
        " if a question is specified."
        " If you are looking to extract specific information from the webpage,"
        " you should specify a question."
    ),
    {
        "url": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The URL to visit",
            required=True,
        ),
        "question": JSONSchema(
            type=JSONSchema.Type.STRING,
            description=(
                "A question that you want to answer using the content of the webpage."
            ),
            required=False,
        ),
    },
)
@validate_url
async def read_webpage(url: str, agent: Agent, question: str = "") -> str:
    """Browse a website and return the answer and links to the user

    Args:
        url (str): The url of the website to browse
        question (str): The question to answer using the content of the webpage

    Returns:
        str: The answer and links to the user and the webdriver
    """
    driver = None
    try:
        # FIXME: agent.config -> something else
        driver = open_page_in_browser(url, agent.legacy_config)

        text = scrape_text_with_selenium(driver)
        links = scrape_links_with_selenium(driver, url)

        return_literal_content = True
        summarized = False
        if not text:
            return f"Website did not contain any text.\n\nLinks: {links}"
        elif (
            agent.llm_provider.count_tokens(text, agent.llm.name)
            > TOKENS_TO_TRIGGER_SUMMARY
        ):
            text = await summarize_memorize_webpage(
                url, text, question or None, agent, driver
            )
            return_literal_content = bool(question)
            summarized = True

        # Limit links to LINKS_TO_RETURN
        if len(links) > LINKS_TO_RETURN:
            links = links[:LINKS_TO_RETURN]

        text_fmt = f"'''{text}'''" if "\n" in text else f"'{text}'"
        links_fmt = "\n".join(f"- {link}" for link in links)
        return (
            f"Page content{' (summary)' if summarized else ''}:"
            if return_literal_content
            else "Answer gathered from webpage:"
        ) + f" {text_fmt}\n\nLinks:\n{links_fmt}"

    except WebDriverException as e:
        # These errors are often quite long and include lots of context.
        # Just grab the first line.
        msg = e.msg.split("\n")[0]
        if "net::" in msg:
            raise BrowsingError(
                "A networking error occurred while trying to load the page: %s"
                % re.sub(r"^unknown error: ", "", msg)
            )
        raise CommandExecutionError(msg)
    finally:
        if driver:
            close_browser(driver)


def scrape_text_with_selenium(driver: WebDriver) -> str:
    """Scrape text from a browser window using selenium

    Args:
        driver (WebDriver): A driver object representing the browser window to scrape

    Returns:
        str: the text scraped from the website
    """

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


def scrape_links_with_selenium(driver: WebDriver, base_url: str) -> list[str]:
    """Scrape links from a website using selenium

    Args:
        driver (WebDriver): A driver object representing the browser window to scrape
        base_url (str): The base URL to use for resolving relative links

    Returns:
        List[str]: The links scraped from the website
    """
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    hyperlinks = extract_hyperlinks(soup, base_url)

    return format_hyperlinks(hyperlinks)


def open_page_in_browser(url: str, config: Config) -> WebDriver:
    """Open a browser window and load a web page using Selenium

    Params:
        url (str): The URL of the page to load
        config (Config): The applicable application configuration

    Returns:
        driver (WebDriver): A driver object representing the browser window to scrape
    """
    logging.getLogger("selenium").setLevel(logging.CRITICAL)

    options_available: dict[str, Type[BrowserOptions]] = {
        "chrome": ChromeOptions,
        "edge": EdgeOptions,
        "firefox": FirefoxOptions,
        "safari": SafariOptions,
    }

    options: BrowserOptions = options_available[config.selenium_web_browser]()
    options.add_argument(f"user-agent={config.user_agent}")

    if config.selenium_web_browser == "firefox":
        if config.selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        driver = FirefoxDriver(
            service=GeckoDriverService(GeckoDriverManager().install()), options=options
        )
    elif config.selenium_web_browser == "edge":
        driver = EdgeDriver(
            service=EdgeDriverService(EdgeDriverManager().install()), options=options
        )
    elif config.selenium_web_browser == "safari":
        # Requires a bit more setup on the users end.
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari  # noqa: E501
        driver = SafariDriver(options=options)
    else:
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if config.selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")

        chromium_driver_path = Path("/usr/bin/chromedriver")

        driver = ChromeDriver(
            service=ChromeDriverService(str(chromium_driver_path))
            if chromium_driver_path.exists()
            else ChromeDriverService(ChromeDriverManager().install()),
            options=options,
        )
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    return driver


def close_browser(driver: WebDriver) -> None:
    """Close the browser

    Args:
        driver (WebDriver): The webdriver to close

    Returns:
        None
    """
    driver.quit()


async def summarize_memorize_webpage(
    url: str,
    text: str,
    question: str | None,
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
        raise ValueError("No text to summarize")

    text_length = len(text)
    logger.info(f"Text length: {text_length} characters")

    # memory = get_memory(agent.legacy_config)

    # new_memory = MemoryItem.from_webpage(
    #     content=text,
    #     url=url,
    #     config=agent.legacy_config,
    #     question=question,
    # )
    # memory.add(new_memory)

    summary, _ = await summarize_text(
        text,
        question=question,
        llm_provider=agent.llm_provider,
        config=agent.legacy_config,  # FIXME
    )
    return summary
