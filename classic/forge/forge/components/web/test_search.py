"""Tests for the modern web search component."""

from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from .search import (
    SearchProvider,
    SearchResult,
    WebSearchComponent,
    WebSearchConfiguration,
)


@pytest.fixture
def web_search_component():
    """Create a WebSearchComponent with no API keys (DDGS multi-engine only)."""
    config = WebSearchConfiguration()
    return WebSearchComponent(config)


@pytest.fixture
def web_search_component_tavily():
    """Create a WebSearchComponent with Tavily configured."""
    config = WebSearchConfiguration(
        tavily_api_key=SecretStr("test-tavily-key"),
    )
    return WebSearchComponent(config)


@pytest.fixture
def web_search_component_serper():
    """Create a WebSearchComponent with Serper configured."""
    config = WebSearchConfiguration(
        serper_api_key=SecretStr("test-serper-key"),
    )
    return WebSearchComponent(config)


@pytest.fixture
def web_search_component_all():
    """Create a WebSearchComponent with all providers configured."""
    config = WebSearchConfiguration(
        tavily_api_key=SecretStr("test-tavily-key"),
        serper_api_key=SecretStr("test-serper-key"),
    )
    return WebSearchComponent(config)


class TestProviderSelection:
    """Test automatic provider selection logic."""

    def test_auto_selects_tavily_when_available(self, web_search_component_tavily):
        assert web_search_component_tavily._get_provider() == SearchProvider.TAVILY

    def test_auto_selects_serper_when_tavily_unavailable(
        self, web_search_component_serper
    ):
        assert web_search_component_serper._get_provider() == SearchProvider.SERPER

    def test_auto_selects_ddgs_when_no_keys(self, web_search_component):
        assert web_search_component._get_provider() == SearchProvider.DDGS

    def test_auto_prefers_tavily_over_serper(self, web_search_component_all):
        assert web_search_component_all._get_provider() == SearchProvider.TAVILY

    def test_explicit_provider_override(self):
        config = WebSearchConfiguration(
            tavily_api_key=SecretStr("test-key"),
            default_provider=SearchProvider.DDGS,
        )
        component = WebSearchComponent(config)
        assert component._get_provider() == SearchProvider.DDGS


class TestDDGSSearch:
    """Test DDGS multi-engine search functionality."""

    @pytest.mark.parametrize(
        "query, num_results, expected_output_parts, return_value",
        [
            (
                "test query",
                3,
                ("Test Result", "https://example.com/test"),
                [
                    {
                        "title": "Test Result",
                        "href": "https://example.com/test",
                        "body": "Test body content",
                    }
                ],
            ),
            ("", 1, (), []),
            ("no results", 1, (), []),
        ],
    )
    def test_ddgs_search(
        self,
        query,
        num_results,
        expected_output_parts,
        return_value,
        mocker,
        web_search_component,
    ):
        mock_ddgs = mocker.patch("forge.components.web.search.DDGS")
        mock_ddgs.return_value.text.return_value = return_value

        result = web_search_component.web_search(query, num_results=num_results)

        for expected in expected_output_parts:
            assert expected in result

    def test_ddgs_tries_multiple_backends_on_failure(
        self, mocker, web_search_component
    ):
        mock_ddgs = mocker.patch("forge.components.web.search.DDGS")
        # Fail twice, succeed on third attempt
        mock_ddgs.return_value.text.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            [
                {
                    "title": "Success",
                    "href": "https://example.com",
                    "body": "Finally worked",
                }
            ],
        ]

        result = web_search_component.web_search("test", num_results=1)
        assert "Success" in result
        assert mock_ddgs.return_value.text.call_count == 3


class TestTavilySearch:
    """Test Tavily search functionality."""

    def test_tavily_search_success(self, mocker, web_search_component_tavily):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer": "This is the AI-generated answer.",
            "results": [
                {
                    "title": "Tavily Result",
                    "url": "https://example.com/tavily",
                    "content": "Tavily content snippet",
                    "score": 0.95,
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()

        mocker.patch("requests.post", return_value=mock_response)

        result = web_search_component_tavily.web_search("test query", num_results=5)

        assert "AI Summary" in result
        assert "AI-generated answer" in result
        assert "Tavily Result" in result
        assert "https://example.com/tavily" in result

    def test_tavily_search_with_content_extraction(
        self, mocker, web_search_component_tavily
    ):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "answer": "Summary answer",
            "results": [
                {
                    "title": "Research Article",
                    "url": "https://example.com/article",
                    "content": "Brief snippet",
                    "score": 0.9,
                    "raw_content": "Full article content with lots of details...",
                }
            ],
        }
        mock_response.raise_for_status = MagicMock()

        mocker.patch("requests.post", return_value=mock_response)

        result = web_search_component_tavily.search_and_extract(
            "research topic", num_results=3
        )

        assert "Research Article" in result
        assert "Full Content:" in result

    def test_tavily_requires_api_key(self, web_search_component):
        # Component without Tavily key should not have search_and_extract command
        commands = list(web_search_component.get_commands())
        command_names = [cmd.names[0] for cmd in commands]
        assert "search_and_extract" not in command_names

    def test_tavily_fallback_to_serper(self, mocker, web_search_component_all):
        # Make Tavily fail
        mock_tavily = mocker.patch.object(
            web_search_component_all,
            "_search_tavily",
            side_effect=Exception("Tavily down"),
        )

        # Mock Serper to succeed
        mock_serper = mocker.patch.object(
            web_search_component_all,
            "_search_serper",
            return_value=[
                SearchResult(
                    title="Serper Result",
                    url="https://example.com/serper",
                    content="Serper fallback content",
                )
            ],
        )

        result = web_search_component_all.web_search("test", num_results=5)

        assert "Serper Result" in result
        mock_tavily.assert_called_once()
        mock_serper.assert_called_once()


class TestSerperSearch:
    """Test Serper search functionality."""

    def test_serper_search_success(self, mocker, web_search_component_serper):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {
                    "title": "Google Result",
                    "link": "https://example.com/google",
                    "snippet": "Google search snippet",
                    "position": 1,
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mocker.patch("requests.post", return_value=mock_response)

        result = web_search_component_serper.web_search("test query", num_results=5)

        assert "Google Result" in result
        assert "https://example.com/google" in result


class TestFallbackBehavior:
    """Test fallback chain when providers fail."""

    def test_full_fallback_to_ddgs(self, mocker, web_search_component_all):
        # Make both Tavily and Serper fail
        mocker.patch.object(
            web_search_component_all,
            "_search_tavily",
            side_effect=Exception("Tavily down"),
        )
        mocker.patch.object(
            web_search_component_all,
            "_search_serper",
            side_effect=Exception("Serper down"),
        )

        # Mock DuckDuckGo to succeed
        mock_ddgs = mocker.patch("forge.components.web.search.DDGS")
        mock_ddgs.return_value.text.return_value = [
            {
                "title": "DDG Fallback",
                "href": "https://example.com/ddg",
                "body": "Fallback content",
            }
        ]

        result = web_search_component_all.web_search("test", num_results=5)

        assert "DDG Fallback" in result

    def test_returns_no_results_message(self, mocker, web_search_component):
        mock_ddgs = mocker.patch("forge.components.web.search.DDGS")
        mock_ddgs.return_value.text.return_value = []

        result = web_search_component.web_search("nonexistent query", num_results=5)

        assert "No search results found" in result


class TestResultFormatting:
    """Test search result formatting."""

    def test_format_results_with_answer(self, web_search_component):
        results = [
            SearchResult(
                title="Test Title",
                url="https://example.com",
                content="Test content",
                score=0.85,
            )
        ]
        formatted = web_search_component._format_results(
            results, answer="AI generated answer"
        )

        assert "## AI Summary" in formatted
        assert "AI generated answer" in formatted
        assert "Test Title" in formatted
        assert "0.85" in formatted

    def test_format_results_with_raw_content(self, web_search_component):
        results = [
            SearchResult(
                title="Article",
                url="https://example.com",
                content="Brief",
                raw_content="Full article text here",
            )
        ]
        formatted = web_search_component._format_results(
            results, include_raw_content=True
        )

        assert "Full Content:" in formatted
        assert "Full article text" in formatted

    def test_format_results_truncates_long_content(self, web_search_component):
        long_content = "x" * 3000
        results = [
            SearchResult(
                title="Long Article",
                url="https://example.com",
                content="Brief",
                raw_content=long_content,
            )
        ]
        formatted = web_search_component._format_results(
            results, include_raw_content=True
        )

        assert "[truncated]" in formatted
        assert len(formatted) < len(long_content) + 500  # Reasonable overhead


class TestLegacyCompatibility:
    """Test backwards compatibility with old API."""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("test string", "test string"),
            (["test1", "test2"], '["test1", "test2"]'),
        ],
    )
    def test_safe_google_results(self, input_val, expected, web_search_component):
        result = web_search_component.safe_google_results(input_val)
        assert result == expected


class TestConfiguration:
    """Test configuration handling."""

    def test_commands_available_based_on_config(self):
        # No keys - only web_search
        config = WebSearchConfiguration()
        component = WebSearchComponent(config)
        commands = list(component.get_commands())
        assert len(commands) == 1
        assert commands[0].names[0] == "web_search"

        # With Tavily key - web_search + search_and_extract
        config = WebSearchConfiguration(tavily_api_key=SecretStr("key"))
        component = WebSearchComponent(config)
        commands = list(component.get_commands())
        assert len(commands) == 2
        command_names = [cmd.names[0] for cmd in commands]
        assert "web_search" in command_names
        assert "search_and_extract" in command_names

    def test_resources_provided(self, web_search_component):
        resources = list(web_search_component.get_resources())
        assert len(resources) == 1
        assert "Internet" in resources[0]
