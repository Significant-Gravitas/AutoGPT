import asyncio
import logging
import re
from pathlib import Path
from sys import platform
from typing import Iterator, Literal, Optional, Type
from urllib.request import urlretrieve

from bs4 import BeautifulSoup
from pydantic import BaseModel
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

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.content_processing.html import extract_hyperlinks, format_hyperlinks
from forge.content_processing.text import extract_information, summarize_text
from forge.llm.providers import MultiProvider
from forge.llm.providers.multi import ModelName
from forge.llm.providers.openai import OpenAIModelName
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError, TooMuchOutputError
from forge.utils.url_validator import validate_url

logger = logging.getLogger(__name__)

FILE_DIR = Path(__file__).parent.parent
MAX_RAW_CONTENT_LENGTH = 500
LINKS_TO_RETURN = 20


BrowserOptions = ChromeOptions | EdgeOptions | FirefoxOptions | SafariOptions


class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""


class WebSeleniumConfiguration(BaseModel):
    llm_name: ModelName = OpenAIModelName.GPT3
    """Name of the llm model used to read websites"""
    web_browser: Literal["chrome", "firefox", "safari", "edge"] = "chrome"
    """Web browser used by Selenium"""
    headless: bool = True
    """Run browser in headless mode"""
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
    )
    """User agent used by the browser"""
    browse_spacy_language_model: str = "en_core_web_sm"
    """Spacy language model used for chunking text"""
    selenium_proxy: Optional[str] = None
    """Http proxy to use with Selenium"""


class WebSeleniumComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[WebSeleniumConfiguration]
):
    """Provides commands to browse the web using Selenium."""

    config_class = WebSeleniumConfiguration

    def __init__(
        self,
        llm_provider: MultiProvider,
        data_dir: Path,
        config: Optional[WebSeleniumConfiguration] = None,
    ):
        ConfigurableComponent.__init__(self, config)
        self.llm_provider = llm_provider
        self.data_dir = data_dir

    def get_resources(self) -> Iterator[str]:
        yield "Ability to read websites."

    def get_commands(self) -> Iterator[Command]:
        yield self.read_webpage

    @command(
        ["read_webpage"],
        (
            "Read a webpage, and extract specific information from it."
            " You must specify either topics_of_interest,"
            " a question, or get_raw_content."
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
                    "A question you want to answer using the content of the webpage."
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
    )
    @validate_url
    async def read_webpage(
        self,
        url: str,
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
            driver = await self.open_page_in_browser(url)

            text = self.scrape_text_with_selenium(driver)
            links = self.scrape_links_with_selenium(driver, url)

            return_literal_content = True
            summarized = False
            if not text:
                return f"Website did not contain any text.\n\nLinks: {links}"
            elif get_raw_content:
                if (
                    output_tokens := self.llm_provider.count_tokens(
                        text, self.config.llm_name
                    )
                ) > MAX_RAW_CONTENT_LENGTH:
                    oversize_factor = round(output_tokens / MAX_RAW_CONTENT_LENGTH, 1)
                    raise TooMuchOutputError(
                        f"Page content is {oversize_factor}x the allowed length "
                        "for `get_raw_content=true`"
                    )
                return text + (f"\n\nLinks: {links}" if links else "")
            else:
                text = await self.summarize_webpage(
                    text, question or None, topics_of_interest
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
            msg = e.msg.split("\n")[0] if e.msg else str(e)
            if "net::" in msg:
                raise BrowsingError(
                    "A networking error occurred while trying to load the page: %s"
                    % re.sub(r"^unknown error: ", "", msg)
                )
            raise CommandExecutionError(msg)
        finally:
            if driver:
                driver.close()

    def scrape_text_with_selenium(self, driver: WebDriver) -> str:
        """Scrape text from a browser window using selenium

        Args:
            driver (WebDriver): A driver object representing
            the browser window to scrape

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

    def scrape_links_with_selenium(self, driver: WebDriver, base_url: str) -> list[str]:
        """Scrape links from a website using selenium

        Args:
            driver (WebDriver): A driver object representing
            the browser window to scrape
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

    async def open_page_in_browser(self, url: str) -> WebDriver:
        """Open a browser window and load a web page using Selenium

        Params:
            url (str): The URL of the page to load
            config (Config): The applicable application configuration

        Returns:
            driver (WebDriver): A driver object representing
            the browser window to scrape
        """
        logging.getLogger("selenium").setLevel(logging.CRITICAL)

        options_available: dict[str, Type[BrowserOptions]] = {
            "chrome": ChromeOptions,
            "edge": EdgeOptions,
            "firefox": FirefoxOptions,
            "safari": SafariOptions,
        }

        options: BrowserOptions = options_available[self.config.web_browser]()
        options.add_argument(f"user-agent={self.config.user_agent}")

        if isinstance(options, FirefoxOptions):
            if self.config.headless:
                options.headless = True  # type: ignore
                options.add_argument("--disable-gpu")
            driver = FirefoxDriver(
                service=GeckoDriverService(GeckoDriverManager().install()),
                options=options,
            )
        elif isinstance(options, EdgeOptions):
            driver = EdgeDriver(
                service=EdgeDriverService(EdgeDriverManager().install()),
                options=options,
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
            if self.config.headless:
                options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")

            if self.config.selenium_proxy:
                options.add_argument(f"--proxy-server={self.config.selenium_proxy}")

            self._sideload_chrome_extensions(options, self.data_dir / "assets" / "crx")

            if (chromium_driver_path := Path("/usr/bin/chromedriver")).exists():
                chrome_service = ChromeDriverService(str(chromium_driver_path))
            else:
                try:
                    chrome_driver = ChromeDriverManager().install()
                except AttributeError as e:
                    if "'NoneType' object has no attribute 'split'" in str(e):
                        # https://github.com/SergeyPirogov/webdriver_manager/issues/649
                        logger.critical(
                            "Connecting to browser failed:"
                            " is Chrome or Chromium installed?"
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

    def _sideload_chrome_extensions(
        self, options: ChromeOptions, dl_folder: Path
    ) -> None:
        crx_download_url_template = "https://clients2.google.com/service/update2/crx?response=redirect&prodversion=99.0&acceptformat=crx3&x=id%3D{crx_id}%26installsource%3Dondemand%26uc"  # noqa
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

    async def summarize_webpage(
        self,
        text: str,
        question: str | None,
        topics_of_interest: list[str],
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

        result = None
        information = None
        if topics_of_interest:
            information = await extract_information(
                text,
                topics_of_interest=topics_of_interest,
                llm_provider=self.llm_provider,
                model_name=self.config.llm_name,
                spacy_model=self.config.browse_spacy_language_model,
            )
            return "\n".join(f"* {i}" for i in information)
        else:
            result, _ = await summarize_text(
                text,
                question=question,
                llm_provider=self.llm_provider,
                model_name=self.config.llm_name,
                spacy_model=self.config.browse_spacy_language_model,
            )
            return result
