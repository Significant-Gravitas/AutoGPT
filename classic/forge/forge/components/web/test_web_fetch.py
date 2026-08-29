"""Tests for the web fetch component."""

from unittest.mock import MagicMock

import httpx
import pytest

from forge.utils.exceptions import CommandExecutionError

from .web_fetch import WebFetchComponent, WebFetchConfiguration


@pytest.fixture
def web_fetch_component():
    """Create a WebFetchComponent with default config."""
    config = WebFetchConfiguration()
    return WebFetchComponent(config)


@pytest.fixture
def sample_html():
    """Sample HTML for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page Title</title>
        <meta name="description" content="Test page description">
        <meta name="author" content="Test Author">
    </head>
    <body>
        <nav>Navigation content to be removed</nav>
        <main>
            <article>
                <h1>Main Article Title</h1>
                <p>This is the main content of the article.</p>
                <p>It contains multiple paragraphs with important information.</p>
                <a href="/relative-link">Relative Link</a>
                <a href="https://example.com/absolute">Absolute Link</a>
            </article>
        </main>
        <footer>Footer content</footer>
        <script>console.log('script to remove');</script>
    </body>
    </html>
    """


class TestFetchWebpage:
    """Test fetch_webpage command."""

    def test_fetch_webpage_extracts_content(
        self, mocker, web_fetch_component, sample_html
    ):
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_response.headers = {"content-length": "1000"}

        mocker.patch.object(
            web_fetch_component.client, "get", return_value=mock_response
        )

        result = web_fetch_component.fetch_webpage(
            "https://example.com/test", include_links=False
        )

        assert "Main Article Title" in result or "main content" in result.lower()
        assert "Title:" in result  # Metadata included

    def test_fetch_webpage_extracts_metadata(
        self, mocker, web_fetch_component, sample_html
    ):
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_response.headers = {}

        mocker.patch.object(
            web_fetch_component.client, "get", return_value=mock_response
        )

        result = web_fetch_component.fetch_webpage(
            "https://example.com/test", include_links=False
        )

        assert "Test Page Title" in result
        assert "Test page description" in result or "Description" in result

    def test_fetch_webpage_extracts_links(
        self, mocker, web_fetch_component, sample_html
    ):
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_response.headers = {}

        mocker.patch.object(
            web_fetch_component.client, "get", return_value=mock_response
        )

        result = web_fetch_component.fetch_webpage(
            "https://example.com/test", include_links=True
        )

        assert "Links" in result
        assert "example.com" in result

    def test_fetch_webpage_handles_timeout(self, mocker, web_fetch_component):
        mocker.patch.object(
            web_fetch_component.client,
            "get",
            side_effect=httpx.TimeoutException("Timeout"),
        )

        with pytest.raises(CommandExecutionError) as exc_info:
            web_fetch_component.fetch_webpage("https://example.com/slow")

        assert "timed out" in str(exc_info.value).lower()

    def test_fetch_webpage_handles_http_error(self, mocker, web_fetch_component):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.reason_phrase = "Not Found"

        mocker.patch.object(
            web_fetch_component.client,
            "get",
            side_effect=httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_response
            ),
        )

        with pytest.raises(CommandExecutionError) as exc_info:
            web_fetch_component.fetch_webpage("https://example.com/missing")

        assert "404" in str(exc_info.value)

    def test_fetch_webpage_respects_max_content_length(
        self, mocker, web_fetch_component
    ):
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "999999999"}

        mocker.patch.object(
            web_fetch_component.client, "get", return_value=mock_response
        )

        with pytest.raises(CommandExecutionError) as exc_info:
            web_fetch_component.fetch_webpage("https://example.com/huge")

        assert "too large" in str(exc_info.value).lower()


class TestFetchRawHtml:
    """Test fetch_raw_html command."""

    def test_fetch_raw_html_returns_html(
        self, mocker, web_fetch_component, sample_html
    ):
        mock_response = MagicMock()
        mock_response.text = sample_html
        mock_response.headers = {}

        mocker.patch.object(
            web_fetch_component.client, "get", return_value=mock_response
        )

        result = web_fetch_component.fetch_raw_html("https://example.com/test")

        assert "<!DOCTYPE html>" in result
        assert "<title>Test Page Title</title>" in result
        assert "<script>" in result  # Raw HTML includes scripts

    def test_fetch_raw_html_truncates_long_content(self, mocker, web_fetch_component):
        long_html = "<html>" + "x" * 100000 + "</html>"

        mock_response = MagicMock()
        mock_response.text = long_html
        mock_response.headers = {}

        mocker.patch.object(
            web_fetch_component.client, "get", return_value=mock_response
        )

        result = web_fetch_component.fetch_raw_html(
            "https://example.com/long", max_length=1000
        )

        assert len(result) < len(long_html)
        assert "truncated" in result.lower()


class TestLinkExtraction:
    """Test link extraction functionality."""

    def test_extracts_absolute_links(self, web_fetch_component):
        html = '<a href="https://example.com/page">Link Text</a>'
        links = web_fetch_component._extract_links(html, "https://base.com")

        assert len(links) == 1
        assert "https://example.com/page" in links[0]
        assert "Link Text" in links[0]

    def test_resolves_relative_links(self, web_fetch_component):
        html = '<a href="/relative/path">Relative</a>'
        links = web_fetch_component._extract_links(html, "https://base.com")

        assert len(links) == 1
        assert "https://base.com/relative/path" in links[0]

    def test_skips_javascript_links(self, web_fetch_component):
        html = """
        <a href="javascript:void(0)">JS Link</a>
        <a href="mailto:test@example.com">Email</a>
        <a href="tel:+1234567890">Phone</a>
        <a href="#section">Anchor</a>
        <a href="https://real.com">Real Link</a>
        """
        links = web_fetch_component._extract_links(html, "https://base.com")

        assert len(links) == 1
        assert "real.com" in links[0]

    def test_respects_max_links(self, web_fetch_component):
        web_fetch_component.config.max_links = 3
        html = "".join(
            f'<a href="https://example.com/{i}">Link {i}</a>' for i in range(10)
        )
        links = web_fetch_component._extract_links(html, "https://base.com")

        assert len(links) == 3


class TestMetadataExtraction:
    """Test metadata extraction functionality."""

    def test_extracts_title(self, web_fetch_component):
        html = "<html><head><title>Page Title</title></head></html>"
        metadata = web_fetch_component._extract_metadata(html)

        assert metadata.get("title") == "Page Title"

    def test_extracts_description(self, web_fetch_component):
        html = '<html><head><meta name="description" content="Page desc"></head></html>'
        metadata = web_fetch_component._extract_metadata(html)

        assert metadata.get("description") == "Page desc"

    def test_extracts_author(self, web_fetch_component):
        html = '<html><head><meta name="author" content="John Doe"></head></html>'
        metadata = web_fetch_component._extract_metadata(html)

        assert metadata.get("author") == "John Doe"

    def test_extracts_og_metadata(self, web_fetch_component):
        html = """
        <html><head>
            <meta property="og:title" content="OG Title">
            <meta property="og:description" content="OG Description">
        </head></html>
        """
        metadata = web_fetch_component._extract_metadata(html)

        assert metadata.get("og_title") == "OG Title"
        assert metadata.get("og_description") == "OG Description"


class TestConfiguration:
    """Test configuration handling."""

    def test_commands_available(self, web_fetch_component):
        commands = list(web_fetch_component.get_commands())
        command_names = [cmd.names[0] for cmd in commands]

        assert "fetch_webpage" in command_names
        assert "fetch_raw_html" in command_names

    def test_resources_provided(self, web_fetch_component):
        resources = list(web_fetch_component.get_resources())
        assert len(resources) == 1
        assert "fetch" in resources[0].lower() or "web" in resources[0].lower()

    def test_custom_config(self):
        config = WebFetchConfiguration(
            timeout=60,
            max_content_length=5_000_000,
            max_links=100,
        )
        component = WebFetchComponent(config)

        assert component.config.timeout == 60
        assert component.config.max_content_length == 5_000_000
        assert component.config.max_links == 100
