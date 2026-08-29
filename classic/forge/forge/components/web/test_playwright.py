"""Tests for the WebPlaywrightComponent."""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from forge.llm.providers.multi import MultiProvider

from . import BrowsingError, WebPlaywrightComponent
from .playwright_browser import WebPlaywrightConfiguration

# Skip all tests if playwright is not installed
pytest.importorskip("playwright")


@pytest.fixture
def web_playwright_component(app_data_dir: Path):
    """Create a WebPlaywrightComponent for testing."""
    return WebPlaywrightComponent(MultiProvider(), app_data_dir)


@pytest.fixture
def web_playwright_component_with_config(app_data_dir: Path):
    """Create a WebPlaywrightComponent with custom config for testing."""
    config = WebPlaywrightConfiguration(
        headless=True,
        max_retries=1,
        page_load_timeout=5000,
    )
    return WebPlaywrightComponent(MultiProvider(), app_data_dir, config=config)


class TestWebPlaywrightConfiguration:
    """Tests for WebPlaywrightConfiguration."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = WebPlaywrightConfiguration()
        assert config.browser_type == "chromium"
        assert config.headless is True
        assert config.max_retries == 3
        assert config.page_load_timeout == 30000
        assert config.max_content_length == 100_000
        assert config.use_cloud_fallback is True
        assert config.block_resources is True

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = WebPlaywrightConfiguration(
            browser_type="firefox",
            headless=False,
            max_retries=5,
            page_load_timeout=60000,
            max_content_length=50_000,
            use_cloud_fallback=False,
            proxy="http://proxy:8080",
        )
        assert config.browser_type == "firefox"
        assert config.headless is False
        assert config.max_retries == 5
        assert config.page_load_timeout == 60000
        assert config.max_content_length == 50_000
        assert config.use_cloud_fallback is False
        assert config.proxy == "http://proxy:8080"


class TestWebPlaywrightComponent:
    """Tests for WebPlaywrightComponent."""

    def test_component_initialization(
        self, web_playwright_component: WebPlaywrightComponent
    ):
        """Test component initializes correctly."""
        assert web_playwright_component._playwright is None
        assert web_playwright_component._browser is None
        assert web_playwright_component._context is None

    def test_get_resources(self, web_playwright_component: WebPlaywrightComponent):
        """Test get_resources returns expected resources."""
        resources = list(web_playwright_component.get_resources())
        assert len(resources) == 1
        assert "Playwright" in resources[0]

    def test_get_commands(self, web_playwright_component: WebPlaywrightComponent):
        """Test get_commands returns expected commands."""
        commands = list(web_playwright_component.get_commands())
        command_names = [cmd.names[0] for cmd in commands]
        assert "read_webpage" in command_names
        assert "take_screenshot" in command_names
        assert "click_element" in command_names
        assert "fill_form" in command_names

    def test_extract_text(self, web_playwright_component: WebPlaywrightComponent):
        """Test text extraction from HTML."""
        html = """
        <html>
            <head><style>.hidden { display: none; }</style></head>
            <body>
                <h1>Hello World</h1>
                <p>This is a test paragraph.</p>
                <script>console.log('ignored');</script>
            </body>
        </html>
        """
        text = web_playwright_component._extract_text(html)
        assert "Hello World" in text
        assert "This is a test paragraph" in text
        assert "console.log" not in text

    def test_extract_links(self, web_playwright_component: WebPlaywrightComponent):
        """Test link extraction from HTML."""
        html = """
        <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="https://example.com/page2">Page 2</a>
            </body>
        </html>
        """
        links = web_playwright_component._extract_links(html, "https://example.com")
        assert len(links) == 2
        assert any("Page 1" in link for link in links)
        assert any("Page 2" in link for link in links)

    def test_truncate_content_short(
        self, web_playwright_component: WebPlaywrightComponent
    ):
        """Test that short content is not truncated."""
        short_text = "This is short text."
        result = web_playwright_component._truncate_content(short_text)
        assert result == short_text
        assert "[Content truncated" not in result

    def test_truncate_content_long(
        self, web_playwright_component: WebPlaywrightComponent
    ):
        """Test that long content is truncated."""
        # Create text longer than max_content_length
        long_text = "x" * (web_playwright_component.config.max_content_length + 1000)
        result = web_playwright_component._truncate_content(long_text)
        assert len(result) < len(long_text)
        assert "[Content truncated" in result


class TestWebPlaywrightComponentAsync:
    """Async tests for WebPlaywrightComponent."""

    @pytest.mark.asyncio
    async def test_browse_website_nonexistent_url(
        self, web_playwright_component_with_config: WebPlaywrightComponent
    ):
        """Test browsing a non-existent URL raises BrowsingError."""
        url = "https://auto-gpt-thinks-this-website-does-not-exist.com"
        question = "How to execute a barrel roll"

        with pytest.raises((BrowsingError, Exception)) as raised:
            await web_playwright_component_with_config.read_webpage(
                url=url, question=question
            )

        # Verify error message is reasonable
        error_msg = str(raised.value)
        assert len(error_msg) < 500

    @pytest.mark.asyncio
    async def test_browse_website_invalid_url(
        self, web_playwright_component: WebPlaywrightComponent
    ):
        """Test browsing an invalid URL raises ValueError."""
        url = "not-a-valid-url"
        question = "What is this page about?"

        with pytest.raises(ValueError, match="Invalid URL format"):
            await web_playwright_component.read_webpage(url=url, question=question)

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, app_data_dir: Path):
        """Test that async context manager properly cleans up resources."""
        component = WebPlaywrightComponent(MultiProvider(), app_data_dir)

        # Mock the cleanup to verify it's called
        component._cleanup = AsyncMock()

        async with component:
            pass

        component._cleanup.assert_called_once()


class TestWebPlaywrightComponentMocked:
    """Tests with mocked browser for faster execution."""

    @pytest.mark.asyncio
    async def test_read_webpage_with_mocked_browser(self, app_data_dir: Path):
        """Test read_webpage with mocked browser."""
        component = WebPlaywrightComponent(MultiProvider(), app_data_dir)

        # Create mocks
        mock_page = AsyncMock()
        mock_page.content.return_value = """
        <html>
            <body>
                <h1>Test Page</h1>
                <p>This is test content.</p>
                <a href="https://example.com">Example Link</a>
            </body>
        </html>
        """
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = AsyncMock()

        mock_playwright = AsyncMock()

        # Set component state as if browser was already initialized
        component._playwright = mock_playwright
        component._browser = mock_browser
        component._context = mock_context

        # Override _smart_wait to not actually wait
        component._smart_wait = AsyncMock()

        result = await component.read_webpage(
            url="https://example.com", get_raw_content=True
        )

        assert "Test Page" in result
        assert "test content" in result
        mock_page.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_take_screenshot_with_mocked_browser(self, app_data_dir: Path):
        """Test take_screenshot with mocked browser."""
        component = WebPlaywrightComponent(MultiProvider(), app_data_dir)

        # Create mocks
        mock_page = AsyncMock()
        mock_page.screenshot = AsyncMock()
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = AsyncMock()

        mock_playwright = AsyncMock()

        # Set component state
        component._playwright = mock_playwright
        component._browser = mock_browser
        component._context = mock_context
        component._smart_wait = AsyncMock()

        result = await component.take_screenshot(
            url="https://example.com", filename="test.png"
        )

        assert "Screenshot saved" in result
        mock_page.screenshot.assert_called_once()
        mock_page.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_click_element_with_mocked_browser(self, app_data_dir: Path):
        """Test click_element with mocked browser."""
        from unittest.mock import MagicMock

        component = WebPlaywrightComponent(MultiProvider(), app_data_dir)

        # Create mocks - locator() is synchronous, but click() is async
        mock_locator = MagicMock()
        mock_locator.click = AsyncMock()

        mock_page = AsyncMock()
        mock_page.locator = MagicMock(return_value=mock_locator)
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()

        # Set component state
        component._playwright = mock_playwright
        component._browser = mock_browser
        component._context = mock_context
        component._smart_wait = AsyncMock()

        result = await component.click_element(
            url="https://example.com", selector="#button"
        )

        assert "Clicked element" in result
        mock_locator.click.assert_called_once()
        mock_page.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_fill_form_with_mocked_browser(self, app_data_dir: Path):
        """Test fill_form with mocked browser."""
        from unittest.mock import MagicMock

        component = WebPlaywrightComponent(MultiProvider(), app_data_dir)

        # Create mocks - locator() is synchronous, but fill() is async
        mock_locator = MagicMock()
        mock_locator.fill = AsyncMock()

        mock_page = AsyncMock()
        mock_page.locator = MagicMock(return_value=mock_locator)
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page.return_value = mock_page

        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()

        # Set component state
        component._playwright = mock_playwright
        component._browser = mock_browser
        component._context = mock_context
        component._smart_wait = AsyncMock()

        result = await component.fill_form(
            url="https://example.com",
            fields={"#username": "testuser", "#password": "testpass"},
        )

        assert "Filled 2 field(s)" in result
        assert mock_locator.fill.call_count == 2
        mock_page.close.assert_called_once()
