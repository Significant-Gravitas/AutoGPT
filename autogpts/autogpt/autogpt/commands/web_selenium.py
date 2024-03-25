"""Commands for browsing a website"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from sys import platform
from typing import TYPE_CHECKING, Optional, Type
from urllib.request import urlretrieve

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

from autogpt.agents.utils.exceptions import CommandExecutionError, TooMuchOutputError
from autogpt.command_decorator import command
from autogpt.core.utils.json_schema import JSONSchema
from autogpt.processing.html import extract_tag_links, format_links
from autogpt.processing.text import extract_information, summarize_text
from autogpt.url_utils.validators import validate_url

COMMAND_CATEGORY = "web_browse"
COMMAND_CATEGORY_TITLE = "Web Browsing"


if TYPE_CHECKING:
    from autogpt.agents.agent import Agent
    from autogpt.config import Config


logger = logging.getLogger(__name__)

FILE_DIR = Path(__file__).parent.parent
MAX_RAW_CONTENT_LENGTH = 500
LINKS_TO_RETURN = 20


class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""


@command(
    "read_webpage",
    (
        "Read a webpage, and extract specific information from it."
        " You must specify either topics_of_interest, a question, or get_raw_content."
    ),
    {
        "url": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="The URL to visit",
            required=True,
        ),
        "topics_of_interest": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            items=JSONSchema(type=JSONSchema.Type.STRING),
            description=(
                "A list of topics about which you want to extract information "
                "from the page."
            ),
            required=False,
        ),
        "question": JSONSchema(
            type=JSONSchema.Type.STRING,
            description=(
                "A question that you want to answer using the content of the webpage."
            ),
            required=False,
        ),
        "get_raw_content": JSONSchema(
            type=JSONSchema.Type.BOOLEAN,
            description=(
                "If true, the unprocessed content of the webpage will be returned. "
                "This consumes a lot of tokens, so use it with caution."
            ),
            required=False,
        ),
    },
    aliases=["browse"],
)
@validate_url
async def read_webpage(
    url: str,
    agent: Agent,
    *,
    topics_of_interest: list[str] = [],
    get_raw_content: bool = False,
    question: str = "",
) -> str:
    """Browse a website and return the answer and links to the user

    Args:
        url (str): The url of the website to browse
        question (str): The question to answer using the content of the webpage

    Returns:
        str: The answer and links to the user and the webdriver
    """
    driver = None
    try:
        driver = await open_page_in_browser(url, agent.legacy_config)

        text = scrape_text(driver)
        links = await scrape_links(url, agent, driver=driver)

        return_literal_content = True
        summarized = False
        if not text:
            return f"Website did not contain any text.\n\nLinks: {links}"
        elif get_raw_content:
            if (
                output_tokens := agent.llm_provider.count_tokens(text, agent.llm.name)
            ) > MAX_RAW_CONTENT_LENGTH:
                oversize_factor = round(output_tokens / MAX_RAW_CONTENT_LENGTH, 1)
                raise TooMuchOutputError(
                    f"Page content is {oversize_factor}x the allowed length "
                    "for `get_raw_content=true`"
                )
            return text + (f"\n\nLinks: {links}" if links else "")
        else:
            text = await summarize_memorize_webpage(
                url, text, question or None, topics_of_interest, agent, driver
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


def scrape_text(driver: WebDriver) -> str:
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


@command(
    "scrape_links",
    "Scrapes a Website for Links",
    {
        "url": JSONSchema(
            type=JSONSchema.Type.STRING, description="The URL to visit", required=True
        ),
        "include_keywords": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            description="Keywords, any of which must appear in the link text",
            required=False,
        ),
        "exclude_keywords": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            description="Keywords, none of which may appear in the link text",
            required=False,
        ),
    },
    aliases=["extract_links", "extract_urls"],
)
async def scrape_links(
    url: str,
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
    return await scrape_tag_links(
        url, "a", agent, include_keywords, exclude_keywords, driver
    )


@command(
    "scrape_image_links",
    "Scrapes a Website for Image Tags and Returns the Image Links",
    {
        "url": JSONSchema(
            type=JSONSchema.Type.STRING, description="The URL to visit", required=True
        ),
        "include_keywords": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            description="Keywords, any of which must appear in the image srcs",
            required=False,
        ),
        "exclude_keywords": JSONSchema(
            type=JSONSchema.Type.ARRAY,
            description="Keywords, none of which may appear in the image srcs",
            required=False,
        ),
    },
    aliases=[
        "extract_images",
        "scrape_images",
        "extract_image_links",
        "extract_image_srcs",
        "scrape_image_sources",
        "scrape_image_srcs",
        "scrape_image_urls",
    ],
)
async def scrape_image_links(
    url: str,
    agent: Agent,
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
    return await scrape_tag_links(
        url, "img", agent, include_keywords, exclude_keywords, driver
    )


@validate_url
async def scrape_tag_links(
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
        driver = await open_page_in_browser(url, agent.legacy_config)

    page_source = driver.execute_script("return document.body.outerHTML;")
    soup = BeautifulSoup(page_source, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    elements = extract_tag_links(
        soup, url, tag_type, include_keywords, exclude_keywords
    )
    return format_links(elements)


async def open_page_in_browser(url: str, config: Config) -> WebDriver:
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

    if isinstance(options, FirefoxOptions):
        if config.selenium_headless:
            options.headless = True
            options.add_argument("--disable-gpu")
        driver = FirefoxDriver(
            service=GeckoDriverService(GeckoDriverManager().install()), options=options
        )
    elif isinstance(options, EdgeOptions):
        driver = EdgeDriver(
            service=EdgeDriverService(EdgeDriverManager().install()), options=options
        )
    elif isinstance(options, SafariOptions):
        # Requires a bit more setup on the users end.
        # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari  # noqa: E501
        driver = SafariDriver(options=options)
    elif isinstance(options, ChromeOptions):
        if platform == "linux" or platform == "linux2":
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--remote-debugging-port=9222")

        options.add_argument("--no-sandbox")
        if config.selenium_headless:
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")

        _sideload_chrome_extensions(options, config.app_data_dir / "assets" / "crx")

        if (chromium_driver_path := Path("/usr/bin/chromedriver")).exists():
            chrome_service = ChromeDriverService(str(chromium_driver_path))
        else:
            try:
                chrome_driver = ChromeDriverManager().install()
            except AttributeError as e:
                if "'NoneType' object has no attribute 'split'" in str(e):
                    # https://github.com/SergeyPirogov/webdriver_manager/issues/649
                    logger.critical(
                        "Connecting to browser failed: is Chrome or Chromium installed?"
                    )
                raise
            chrome_service = ChromeDriverService(chrome_driver)
        driver = ChromeDriver(service=chrome_service, options=options)

    driver.get(url)

    # Wait for page to be ready, sleep 2 seconds, wait again until page ready.
    # This allows the cookiewall squasher time to get rid of cookie walls.
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    await asyncio.sleep(2)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    return driver


def _sideload_chrome_extensions(options: ChromeOptions, dl_folder: Path) -> None:
    crx_download_url_template = "https://clients2.google.com/service/update2/crx?response=redirect&prodversion=49.0&acceptformat=crx3&x=id%3D{crx_id}%26installsource%3Dondemand%26uc"  # noqa
    cookiewall_squasher_crx_id = "edibdbjcniadpccecjdfdjjppcpchdlm"
    adblocker_crx_id = "cjpalhdlnbpafiamejdnhcphjbkeiagm"

    # Make sure the target folder exists
    dl_folder.mkdir(parents=True, exist_ok=True)

    for crx_id in (cookiewall_squasher_crx_id, adblocker_crx_id):
        crx_path = dl_folder / f"{crx_id}.crx"
        if not crx_path.exists():
            logger.debug(f"Downloading CRX {crx_id}...")
            crx_download_url = crx_download_url_template.format(crx_id=crx_id)
            urlretrieve(crx_download_url, crx_path)
            logger.debug(f"Downloaded {crx_path.name}")
        options.add_extension(str(crx_path))


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
    topics_of_interest: list[str],
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
    logger.debug(f"Web page content length: {text_length} characters")

    # memory = get_memory(agent.legacy_config)

    # new_memory = MemoryItem.from_webpage(
    #     content=text,
    #     url=url,
    #     config=agent.legacy_config,
    #     question=question,
    # )
    # memory.add(new_memory)

    result = None
    information = None
    if topics_of_interest:
        information = await extract_information(
            text,
            topics_of_interest=topics_of_interest,
            llm_provider=agent.llm_provider,
            config=agent.legacy_config,
        )
        return "\n".join(f"* {i}" for i in information)
    else:
        result, _ = await summarize_text(
            text,
            question=question,
            llm_provider=agent.llm_provider,
            config=agent.legacy_config,
        )
        return result
