"""
Lightweight web page fetching component.

Provides fast HTTP-based page fetching without browser overhead.
Uses trafilatura for intelligent content extraction.
"""

import logging
from typing import Iterator, Literal, Optional
from urllib.parse import urljoin

import httpx
import trafilatura
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError
from forge.utils.url_validator import validate_url

logger = logging.getLogger(__name__)

# Default headers to mimic a browser
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


class WebFetchConfiguration(BaseModel):
    """Configuration for the web fetch component."""

    timeout: int = 30
    """Request timeout in seconds"""

    max_content_length: int = 10_000_000  # 10MB
    """Maximum content length to download"""

    follow_redirects: bool = True
    """Whether to follow HTTP redirects"""

    extract_links: bool = True
    """Whether to extract links from pages"""

    max_links: int = 50
    """Maximum number of links to return"""

    include_metadata: bool = True
    """Whether to include page metadata (title, description, etc.)"""


class WebFetchComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[WebFetchConfiguration]
):
    """
    Lightweight web page fetching component.

    Provides fast HTTP-based page fetching without browser overhead.
    Uses trafilatura for intelligent main content extraction.
    """

    config_class = WebFetchConfiguration

    def __init__(self, config: Optional[WebFetchConfiguration] = None):
        ConfigurableComponent.__init__(self, config)
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-loaded HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                headers=DEFAULT_HEADERS,
                timeout=self.config.timeout,
                follow_redirects=self.config.follow_redirects,
            )
        return self._client

    def get_resources(self) -> Iterator[str]:
        yield "Ability to fetch and extract content from web pages."

    def get_commands(self) -> Iterator[Command]:
        yield self.fetch_webpage
        yield self.fetch_raw_html

    def _fetch_url(self, url: str) -> httpx.Response:
        """Fetch a URL and return the response."""
        try:
            response = self.client.get(url)
            response.raise_for_status()

            # Check content length
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self.config.max_content_length:
                raise CommandExecutionError(
                    f"Content too large: {int(content_length)} bytes "
                    f"(max: {self.config.max_content_length})"
                )

            return response

        except httpx.TimeoutException:
            raise CommandExecutionError(
                f"Request timed out after {self.config.timeout} seconds"
            )
        except httpx.HTTPStatusError as e:
            raise CommandExecutionError(
                f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
            )
        except httpx.RequestError as e:
            raise CommandExecutionError(f"Request failed: {e}")

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract links from HTML content."""
        soup = BeautifulSoup(html, "html.parser")
        links = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Skip javascript, mailto, tel links
            if href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue

            # Resolve relative URLs
            absolute_url = urljoin(base_url, href)

            # Only include http(s) links
            if absolute_url.startswith(("http://", "https://")):
                # Get link text
                text = a.get_text(strip=True)[:100] or "[no text]"
                links.append(f"{text}: {absolute_url}")

                if len(links) >= self.config.max_links:
                    break

        return links

    def _extract_metadata(self, html: str) -> dict[str, str]:
        """Extract metadata from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        metadata = {}

        # Title
        title_tag = soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text(strip=True)

        # Meta description
        desc = soup.find("meta", attrs={"name": "description"})
        if isinstance(desc, Tag) and desc.get("content"):
            metadata["description"] = str(desc["content"])

        # Open Graph title/description
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if isinstance(og_title, Tag) and og_title.get("content"):
            metadata["og_title"] = str(og_title["content"])

        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if isinstance(og_desc, Tag) and og_desc.get("content"):
            metadata["og_description"] = str(og_desc["content"])

        # Author
        author = soup.find("meta", attrs={"name": "author"})
        if isinstance(author, Tag) and author.get("content"):
            metadata["author"] = str(author["content"])

        # Published date
        for attr in ["article:published_time", "datePublished", "date"]:
            date_tag = soup.find("meta", attrs={"property": attr}) or soup.find(
                "meta", attrs={"name": attr}
            )
            if isinstance(date_tag, Tag) and date_tag.get("content"):
                metadata["published"] = str(date_tag["content"])
                break

        return metadata

    @command(
        ["fetch_webpage", "fetch", "download_page"],
        "Fetch a webpage and extract its main content as clean text. "
        "Much faster than read_webpage (no browser needed).",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL to fetch",
                required=True,
            ),
            "output_format": JSONSchema(
                type=JSONSchema.Type.STRING,
                description=(
                    "Output format: 'text' (plain text), 'markdown' (with formatting), "
                    "or 'xml' (structured). Default: 'markdown'"
                ),
                required=False,
            ),
            "include_links": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Whether to include extracted links. Default: true",
                required=False,
            ),
            "include_comments": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description="Whether to include page comments. Default: false",
                required=False,
            ),
        },
    )
    @validate_url
    def fetch_webpage(
        self,
        url: str,
        output_format: Literal["text", "markdown", "xml"] = "markdown",
        include_links: bool = True,
        include_comments: bool = False,
    ) -> str:
        """
        Fetch a webpage and extract its main content.

        Uses trafilatura for intelligent content extraction - automatically
        removes navigation, ads, boilerplate, and extracts the main article text.

        Args:
            url: The URL to fetch
            output_format: Output format (text, markdown, xml)
            include_links: Whether to include links from the page
            include_comments: Whether to include comments section

        Returns:
            Extracted content with optional metadata and links
        """
        response = self._fetch_url(url)
        html = response.text

        # Extract main content using trafilatura
        extract_kwargs = {
            "include_comments": include_comments,
            "include_links": output_format == "markdown",
            "include_images": False,
            "include_tables": True,
            "no_fallback": False,
        }

        if output_format == "markdown":
            content = trafilatura.extract(
                html,
                output_format="markdown",
                **extract_kwargs,  # type: ignore[arg-type]
            )
        elif output_format == "xml":
            content = trafilatura.extract(
                html,
                output_format="xml",
                **extract_kwargs,  # type: ignore[arg-type]
            )
        else:
            content = trafilatura.extract(
                html, **extract_kwargs  # type: ignore[arg-type]
            )

        if not content:
            # Fallback to basic BeautifulSoup extraction
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = soup.get_text(separator="\n", strip=True)
            if not content:
                return "Could not extract content from this page."

        # Build output
        output_parts = []

        # Add metadata
        if self.config.include_metadata:
            metadata = self._extract_metadata(html)
            if metadata:
                meta_lines = []
                if "title" in metadata:
                    meta_lines.append(f"**Title:** {metadata['title']}")
                if "description" in metadata:
                    meta_lines.append(f"**Description:** {metadata['description']}")
                if "author" in metadata:
                    meta_lines.append(f"**Author:** {metadata['author']}")
                if "published" in metadata:
                    meta_lines.append(f"**Published:** {metadata['published']}")
                if meta_lines:
                    output_parts.append("## Page Info\n" + "\n".join(meta_lines))

        # Add main content
        output_parts.append(f"## Content\n{content}")

        # Add links
        if include_links and self.config.extract_links:
            links = self._extract_links(html, url)
            if links:
                links_text = "\n".join(f"- {link}" for link in links)
                output_parts.append(f"## Links ({len(links)})\n{links_text}")

        return "\n\n".join(output_parts)

    @command(
        ["fetch_raw_html", "get_html"],
        "Fetch a webpage and return the raw HTML. Use this when you need "
        "to inspect the page structure or extract specific elements.",
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL to fetch",
                required=True,
            ),
            "max_length": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Maximum characters to return. Default: 50000",
                required=False,
            ),
        },
    )
    @validate_url
    def fetch_raw_html(self, url: str, max_length: int = 50000) -> str:
        """
        Fetch a webpage and return the raw HTML.

        Args:
            url: The URL to fetch
            max_length: Maximum characters to return

        Returns:
            Raw HTML content (truncated if necessary)
        """
        response = self._fetch_url(url)
        html = response.text

        if len(html) > max_length:
            return html[:max_length] + f"\n\n... [truncated, {len(html)} total chars]"

        return html
