"""Web browsing component using Playwright for reliable browser automation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from bs4 import BeautifulSoup
from pydantic import BaseModel, SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.content_processing.html import extract_hyperlinks, format_hyperlinks
from forge.content_processing.text import extract_information, summarize_text
from forge.llm.providers import MultiProvider
from forge.llm.providers.multi import ModelName
from forge.llm.providers.openai import OpenAIModelName
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError
from forge.utils.url_validator import validate_url

logger = logging.getLogger(__name__)

# Lazy imports for playwright to avoid import errors if not installed
Playwright = None
Browser = None
Page = None
BrowserContext = None
PlaywrightError = None

MAX_RAW_CONTENT_LENGTH = 500
LINKS_TO_RETURN = 20
MAX_CONTENT_LENGTH = 100_000


def _ensure_playwright_imported():
    """Lazily import playwright to provide better error messages."""
    global Playwright, Browser, Page, BrowserContext, PlaywrightError
    if Playwright is None:
        try:
            from playwright.async_api import Browser as _Browser
            from playwright.async_api import BrowserContext as _BrowserContext
            from playwright.async_api import Error as _PlaywrightError
            from playwright.async_api import Page as _Page
            from playwright.async_api import Playwright as _Playwright

            Playwright = _Playwright
            Browser = _Browser
            Page = _Page
            BrowserContext = _BrowserContext
            PlaywrightError = _PlaywrightError
        except ImportError:
            raise ImportError(
                "Playwright is not installed. Install it with: "
                "poetry install && playwright install chromium"
            )


class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""


class WebPlaywrightConfiguration(BaseModel):
    """Configuration for the Playwright-based web browsing component."""

    llm_name: ModelName = OpenAIModelName.GPT3
    """Name of the LLM model used to read websites"""

    browser_type: Literal["chromium", "firefox", "webkit"] = "chromium"
    """Browser engine to use"""

    headless: bool = True
    """Run browser in headless mode"""

    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    """User agent string for the browser"""

    browse_spacy_language_model: str = "en_core_web_sm"
    """Spacy language model used for chunking text"""

    max_retries: int = 3
    """Maximum number of retry attempts for transient failures"""

    retry_delay: float = 1.0
    """Base delay in seconds between retries (exponential backoff)"""

    page_load_timeout: int = 30000
    """Timeout in milliseconds for page loads"""

    max_content_length: int = MAX_CONTENT_LENGTH
    """Maximum content length before truncation (characters)"""

    # Optional cloud fallback via CDP
    browserless_token: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="BROWSERLESS_TOKEN"
    )
    """Token for Browserless.io cloud browser service (optional)"""

    use_cloud_fallback: bool = True
    """Whether to fallback to cloud browser if local browser fails"""

    proxy: Optional[str] = None
    """HTTP proxy to use (e.g., http://proxy:8080)"""

    block_resources: bool = True
    """Block images, fonts, and other non-essential resources for faster loads"""


class WebPlaywrightComponent(
    DirectiveProvider,
    CommandProvider,
    ConfigurableComponent[WebPlaywrightConfiguration],
):
    """Provides commands to browse the web using Playwright.

    Features over Selenium:
    - Connection pooling: Single browser instance reused across commands
    - Smart waiting: Adaptive waits instead of hardcoded sleeps
    - Retry with backoff: Automatic retries on transient failures
    - Content truncation: Large pages are truncated instead of rejected
    - Proper cleanup: Browser properly closed on exit
    - Cloud fallback: Optional connection to Browserless.io if local fails
    """

    config_class = WebPlaywrightConfiguration

    def __init__(
        self,
        llm_provider: MultiProvider,
        data_dir: Path,
        config: Optional[WebPlaywrightConfiguration] = None,
    ):
        ConfigurableComponent.__init__(self, config)
        self.llm_provider = llm_provider
        self.data_dir = data_dir
        self._playwright = None
        self._browser = None
        self._context = None

    async def __aenter__(self):
        """Async context manager entry - initializes browser."""
        await self._ensure_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleans up browser."""
        await self._cleanup()

    async def _ensure_browser(self) -> None:
        """Lazily initialize the browser if not already initialized."""
        _ensure_playwright_imported()

        if self._browser is not None:
            return

        from playwright.async_api import async_playwright

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._launch_browser()
            self._context = await self._create_context()
            logger.debug("Playwright browser initialized successfully")
        except Exception as e:
            logger.warning(f"Local browser launch failed: {e}")
            if self.config.use_cloud_fallback and self.config.browserless_token:
                await self._connect_cloud_browser()
            else:
                raise BrowsingError(
                    f"Failed to launch browser: {e}. "
                    "Run 'playwright install chromium' to install browser binaries."
                )

    async def _launch_browser(self):
        """Launch a local browser instance."""
        browser_launcher = getattr(self._playwright, self.config.browser_type)

        launch_args: dict[str, Any] = {
            "headless": self.config.headless,
        }

        # Add proxy if configured
        if self.config.proxy:
            launch_args["proxy"] = {"server": self.config.proxy}

        return await browser_launcher.launch(**launch_args)

    async def _connect_cloud_browser(self) -> None:
        """Connect to Browserless.io cloud browser service."""
        if not self.config.browserless_token:
            raise BrowsingError("No browserless token configured for cloud fallback")

        _ensure_playwright_imported()
        from playwright.async_api import async_playwright

        token = self.config.browserless_token.get_secret_value()
        ws_endpoint = f"wss://chrome.browserless.io?token={token}"

        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.connect_over_cdp(
                ws_endpoint
            )
            self._context = await self._create_context()
            logger.info("Connected to Browserless.io cloud browser")
        except Exception as e:
            raise BrowsingError(f"Failed to connect to cloud browser: {e}")

    async def _create_context(self):
        """Create a browser context with configured settings."""
        assert self._browser is not None, "Browser not initialized"
        context = await self._browser.new_context(
            user_agent=self.config.user_agent,
            viewport={"width": 1920, "height": 1080},
        )

        # Block non-essential resources for faster page loads
        if self.config.block_resources:
            await context.route(
                "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,eot}",
                lambda route: route.abort(),
            )

        return context

    async def _cleanup(self) -> None:
        """Clean up browser resources properly."""
        try:
            if self._context:
                await self._context.close()
                self._context = None
            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            logger.debug("Playwright browser cleaned up")
        except Exception as e:
            logger.warning(f"Error during browser cleanup: {e}")

    async def _smart_wait(self, page) -> None:
        """Wait for page to be fully loaded using adaptive waiting.

        This replaces hardcoded sleeps with intelligent waiting:
        1. Wait for network to be idle (no requests for 500ms)
        2. Wait for DOM to stabilize (no mutations for 500ms)
        """
        try:
            # Wait for network idle
            await page.wait_for_load_state(
                "networkidle", timeout=self.config.page_load_timeout
            )
        except Exception:
            # Fallback to domcontentloaded if networkidle times out
            logger.debug("Network idle timeout, using domcontentloaded instead")
            await page.wait_for_load_state("domcontentloaded")

        # Wait for DOM to stabilize
        try:
            await page.evaluate(
                """
                () => new Promise(resolve => {
                    let timer;
                    const observer = new MutationObserver(() => {
                        clearTimeout(timer);
                        timer = setTimeout(() => {
                            observer.disconnect();
                            resolve();
                        }, 500);
                    });
                    observer.observe(document.body, {
                        childList: true,
                        subtree: true
                    });
                    timer = setTimeout(() => {
                        observer.disconnect();
                        resolve();
                    }, 500);
                })
            """
            )
        except Exception as e:
            logger.debug(f"DOM stability check failed (non-critical): {e}")

    async def _open_page(self, url: str):
        """Open a new page and navigate to URL with smart waiting."""
        await self._ensure_browser()
        assert self._context is not None, "Browser context not initialized"

        page = await self._context.new_page()
        try:
            await page.goto(url, timeout=self.config.page_load_timeout)
            await self._smart_wait(page)
            return page
        except Exception as e:
            await page.close()
            raise e

    def _extract_text(self, html: str) -> str:
        """Extract clean text from HTML content."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "noscript"]):
            element.extract()

        # Get text and clean it up
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract and format links from HTML content."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.extract()

        hyperlinks = extract_hyperlinks(soup, base_url)
        return format_hyperlinks(hyperlinks)

    def _truncate_content(self, text: str) -> str:
        """Truncate content if it exceeds max length."""
        if len(text) > self.config.max_content_length:
            truncated = text[: self.config.max_content_length]
            return f"{truncated}\n\n[Content truncated - {len(text)} chars total]"
        return text

    def get_resources(self) -> Iterator[str]:
        yield "Ability to read websites using Playwright browser automation."

    def get_commands(self) -> Iterator[Command]:
        yield self.read_webpage
        yield self.take_screenshot
        yield self.click_element
        yield self.fill_form

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _read_webpage_with_retry(
        self,
        url: str,
        topics_of_interest: list[str],
        get_raw_content: bool,
        question: str,
    ) -> str:
        """Internal method with retry logic for read_webpage."""
        page = None
        try:
            page = await self._open_page(url)

            html = await page.content()
            text = self._extract_text(html)
            links = self._extract_links(html, url)

            return_literal_content = True
            summarized = False

            if not text:
                return f"Website did not contain any text.\n\nLinks: {links}"
            elif get_raw_content:
                # Truncate instead of rejecting large pages
                text = self._truncate_content(text)
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

        finally:
            if page:
                await page.close()

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
                    "Large pages will be truncated to avoid overwhelming context."
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
        """Browse a website and return the answer and links to the user.

        Args:
            url: The url of the website to browse
            topics_of_interest: Topics to extract information about
            get_raw_content: If true, return raw page content (truncated if large)
            question: The question to answer using the content of the webpage

        Returns:
            The answer and links to the user
        """
        _ensure_playwright_imported()
        try:
            return await self._read_webpage_with_retry(
                url, topics_of_interest, get_raw_content, question
            )
        except Exception as e:
            error_msg = str(e)
            if "net::" in error_msg:
                raise BrowsingError(
                    "A networking error occurred while trying to load the page: "
                    f"{error_msg}"
                )
            raise CommandExecutionError(f"Failed to read webpage: {error_msg}")

    async def summarize_webpage(
        self,
        text: str,
        question: str | None,
        topics_of_interest: list[str],
    ) -> str:
        """Summarize text using the LLM.

        Args:
            text: The text to summarize
            question: The question to ask the model
            topics_of_interest: Topics to extract information about

        Returns:
            The summary of the text
        """
        if not text:
            raise ValueError("No text to summarize")

        text_length = len(text)
        logger.debug(f"Web page content length: {text_length} characters")

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

    @command(
        ["take_screenshot"],
        "Take a screenshot of a webpage and save it to a file.",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL of the webpage to screenshot",
                required=True,
            ),
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Filename for screenshot (e.g. 'screenshot.png')",
                required=True,
            ),
            "full_page": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Capture full page including scrollable content",
                required=False,
            ),
        },
    )
    @validate_url
    async def take_screenshot(
        self, url: str, filename: str, full_page: bool = False
    ) -> str:
        """Take a screenshot of a webpage.

        Args:
            url: The URL to screenshot
            filename: The filename to save to
            full_page: Whether to capture full scrollable page

        Returns:
            Success message with file path
        """
        _ensure_playwright_imported()
        page = None
        try:
            page = await self._open_page(url)

            # Save screenshot
            screenshot_path = self.data_dir / filename
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)

            await page.screenshot(path=str(screenshot_path), full_page=full_page)

            return f"Screenshot saved to {screenshot_path}"

        except Exception as e:
            raise CommandExecutionError(f"Screenshot failed: {e}")
        finally:
            if page:
                await page.close()

    @command(
        ["click_element"],
        "Click an element on a webpage identified by a CSS selector or XPath.",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL of the webpage",
                required=True,
            ),
            "selector": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="CSS selector or XPath expression to find the element",
                required=True,
            ),
            "selector_type": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Type of selector: 'css' or 'xpath' (default: 'css')",
                required=False,
            ),
            "timeout": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Timeout in seconds to wait for element (default: 10)",
                required=False,
            ),
        },
    )
    @validate_url
    async def click_element(
        self,
        url: str,
        selector: str,
        selector_type: str = "css",
        timeout: int = 10,
    ) -> str:
        """Click an element on a webpage.

        Args:
            url: The URL of the webpage
            selector: The CSS selector or XPath
            selector_type: Type of selector ('css' or 'xpath')
            timeout: Timeout to wait for element

        Returns:
            Success message
        """
        _ensure_playwright_imported()
        page = None
        try:
            page = await self._open_page(url)

            # Convert timeout to milliseconds
            timeout_ms = timeout * 1000

            # Use appropriate locator based on selector type
            if selector_type == "xpath":
                locator = page.locator(f"xpath={selector}")
            else:
                locator = page.locator(selector)

            # Wait for element and click
            await locator.click(timeout=timeout_ms)

            # Wait for any navigation or changes
            await self._smart_wait(page)

            return f"Clicked element matching '{selector}'"

        except Exception as e:
            raise CommandExecutionError(f"Click failed: {e}")
        finally:
            if page:
                await page.close()

    @command(
        ["fill_form"],
        "Fill form fields on a webpage with provided values.",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL of the webpage with the form",
                required=True,
            ),
            "fields": JSONSchema(
                type=JSONSchema.Type.OBJECT,
                description="Dictionary mapping CSS selectors to values to enter",
                required=True,
            ),
            "submit": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Whether to submit the form after filling (default: False)",
                required=False,
            ),
        },
    )
    @validate_url
    async def fill_form(
        self,
        url: str,
        fields: dict[str, str],
        submit: bool = False,
    ) -> str:
        """Fill form fields on a webpage.

        Args:
            url: The URL of the webpage
            fields: Dict mapping selectors to values
            submit: Whether to submit the form

        Returns:
            Success message with filled fields
        """
        _ensure_playwright_imported()
        page = None
        try:
            page = await self._open_page(url)

            filled = []
            for selector, value in fields.items():
                try:
                    locator = page.locator(selector)
                    await locator.fill(value)
                    filled.append(selector)
                except Exception as e:
                    raise CommandExecutionError(
                        f"Could not fill field '{selector}': {e}"
                    )

            if submit and filled:
                # Try to find and click submit button
                try:
                    submit_btn = page.locator(
                        "button[type='submit'], input[type='submit']"
                    )
                    await submit_btn.click()
                    await self._smart_wait(page)
                except Exception:
                    # Try submitting the form directly
                    try:
                        await page.locator("form").evaluate("form => form.submit()")
                        await self._smart_wait(page)
                    except Exception as e:
                        raise CommandExecutionError(f"Could not submit form: {e}")

            msg = f"Filled {len(filled)} field(s): {', '.join(filled)}"
            if submit:
                msg += " and submitted form"
            return msg

        except CommandExecutionError:
            raise
        except Exception as e:
            raise CommandExecutionError(f"Form fill failed: {e}")
        finally:
            if page:
                await page.close()

    async def close(self) -> None:
        """Explicitly close the browser and clean up resources.

        Call this when done using the component to ensure proper cleanup.
        """
        await self._cleanup()
