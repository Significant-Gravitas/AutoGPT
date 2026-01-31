"""
Modern web search component with tiered provider support.

Provider hierarchy:
1. Tavily (primary) - AI-optimized results with content extraction
2. Serper (secondary) - Fast, cheap Google SERP results
3. DDGS (fallback) - Free multi-engine search (DuckDuckGo, Bing, Brave, Google, etc.)
"""

import json
import logging
from enum import Enum
from typing import Iterator, Literal, Optional

import requests
from ddgs import DDGS
from pydantic import BaseModel, SecretStr

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.models.config import UserConfigurable
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Available backends for DDGS text search
# Ordered by reliability/quality for fallback chain
DDGS_BACKENDS = [
    "duckduckgo",
    "bing",
    "brave",
    "google",
    "mojeek",
    "yahoo",
    "yandex",
    "wikipedia",
]


class SearchProvider(str, Enum):
    """Available search providers."""

    TAVILY = "tavily"
    SERPER = "serper"
    DDGS = "ddgs"  # Multi-engine free search
    AUTO = "auto"  # Automatic provider selection based on availability


class SearchResult(BaseModel):
    """Standardized search result format."""

    title: str
    url: str
    content: str  # Snippet or extracted content
    score: Optional[float] = None  # Relevance score if available
    raw_content: Optional[str] = None  # Full page content if extracted


class WebSearchConfiguration(BaseModel):
    """Configuration for the web search component."""

    # Tavily settings (primary provider)
    tavily_api_key: Optional[SecretStr] = UserConfigurable(
        None, from_env="TAVILY_API_KEY", exclude=True
    )
    tavily_search_depth: Literal["basic", "advanced"] = "basic"
    tavily_include_answer: bool = True  # Get AI-generated answer
    tavily_include_raw_content: bool = False  # Extract full page content

    # Serper settings (secondary provider)
    serper_api_key: Optional[SecretStr] = UserConfigurable(
        None, from_env="SERPER_API_KEY", exclude=True
    )

    # DDGS settings (free fallback with multiple backends)
    ddgs_backend: Literal[
        "auto",
        "duckduckgo",
        "bing",
        "brave",
        "google",
        "mojeek",
        "yahoo",
        "yandex",
        "wikipedia",
    ] = "auto"
    ddgs_region: str = "us-en"  # Region for localized results
    ddgs_safesearch: Literal["on", "moderate", "off"] = "moderate"

    # General settings
    default_provider: SearchProvider = SearchProvider.AUTO
    max_results: int = 8

    # Legacy settings (deprecated)
    google_api_key: Optional[SecretStr] = UserConfigurable(
        None, from_env="GOOGLE_API_KEY", exclude=True
    )
    google_custom_search_engine_id: Optional[SecretStr] = UserConfigurable(
        None, from_env="GOOGLE_CUSTOM_SEARCH_ENGINE_ID", exclude=True
    )
    # Legacy aliases for backwards compatibility
    duckduckgo_max_attempts: int = 3  # Now used as max backend attempts
    duckduckgo_backend: Literal["api", "html", "lite"] = (
        "api"  # Ignored, use ddgs_backend
    )


class WebSearchComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[WebSearchConfiguration]
):
    """
    Modern web search component with tiered provider support.

    Provides intelligent web search with automatic provider selection:
    - Tavily: AI-optimized results with optional content extraction
    - Serper: Fast Google SERP results at low cost
    - DDGS: Free multi-engine fallback (DuckDuckGo, Bing, Brave, Google, etc.)
    """

    config_class = WebSearchConfiguration

    def __init__(self, config: Optional[WebSearchConfiguration] = None):
        ConfigurableComponent.__init__(self, config)
        self._ddgs_client: "Optional[DDGS]" = None  # type: ignore[type-arg]
        self._log_provider_status()

    def _log_provider_status(self) -> None:
        """Log which providers are available."""
        providers = []
        if self.config.tavily_api_key:
            providers.append("Tavily (primary)")
        if self.config.serper_api_key:
            providers.append("Serper (secondary)")
        providers.append("DDGS multi-engine (fallback)")

        logger.info(f"Web search providers available: {', '.join(providers)}")

        if not self.config.tavily_api_key and not self.config.serper_api_key:
            logger.info(
                "No premium search API keys configured. "
                "Using DDGS multi-engine search (free). "
                "Set TAVILY_API_KEY or SERPER_API_KEY for enhanced results."
            )

    @property
    def ddgs_client(self) -> "DDGS":  # type: ignore[type-arg]
        """Lazy-loaded DDGS client."""
        if self._ddgs_client is None:
            self._ddgs_client = DDGS()
        return self._ddgs_client

    def get_resources(self) -> Iterator[str]:
        yield "Internet access for searches and information gathering."

    def get_commands(self) -> Iterator[Command]:
        yield self.web_search
        if self.config.tavily_api_key:
            yield self.search_and_extract

    def _get_provider(self) -> SearchProvider:
        """Determine which provider to use based on configuration."""
        if self.config.default_provider != SearchProvider.AUTO:
            return self.config.default_provider

        # Auto-select: prefer Tavily > Serper > DDGS
        if self.config.tavily_api_key:
            return SearchProvider.TAVILY
        elif self.config.serper_api_key:
            return SearchProvider.SERPER
        else:
            return SearchProvider.DDGS

    def _search_tavily(
        self,
        query: str,
        num_results: int,
        include_answer: bool = True,
        include_raw_content: bool = False,
        search_depth: Optional[str] = None,
    ) -> tuple[list[SearchResult], Optional[str]]:
        """
        Search using Tavily API.

        Returns:
            Tuple of (results list, AI-generated answer or None)
        """
        if not self.config.tavily_api_key:
            raise ConfigurationError("Tavily API key not configured")

        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}

        payload = {
            "api_key": self.config.tavily_api_key.get_secret_value(),
            "query": query,
            "max_results": num_results,
            "search_depth": search_depth or self.config.tavily_search_depth,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=r.get("score"),
                    raw_content=r.get("raw_content") if include_raw_content else None,
                )
                for r in data.get("results", [])
            ]

            answer = data.get("answer") if include_answer else None
            return results, answer

        except requests.RequestException as e:
            logger.error(f"Tavily search failed: {e}")
            raise

    def _search_serper(self, query: str, num_results: int) -> list[SearchResult]:
        """Search using Serper.dev API (Google SERP)."""
        if not self.config.serper_api_key:
            raise ConfigurationError("Serper API key not configured")

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.config.serper_api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for r in data.get("organic", []):
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("link", ""),
                        content=r.get("snippet", ""),
                        score=r.get("position"),  # Position as pseudo-score
                    )
                )

            return results

        except requests.RequestException as e:
            logger.error(f"Serper search failed: {e}")
            raise

    def _search_ddgs(self, query: str, num_results: int) -> list[SearchResult]:
        """
        Search using DDGS multi-engine search.

        Tries multiple backends in order until one succeeds:
        DuckDuckGo -> Bing -> Brave -> Google -> Mojeek -> Yahoo -> Yandex
        """
        if not query:
            return []

        # Determine which backends to try
        if self.config.ddgs_backend == "auto":
            backends_to_try = DDGS_BACKENDS.copy()
        else:
            # Put configured backend first, then others as fallback
            backends_to_try = [self.config.ddgs_backend] + [
                b for b in DDGS_BACKENDS if b != self.config.ddgs_backend
            ]

        max_attempts = min(self.config.duckduckgo_max_attempts, len(backends_to_try))
        last_error: Optional[Exception] = None

        for backend in backends_to_try[:max_attempts]:
            try:
                logger.debug(f"Trying DDGS backend: {backend}")
                raw_results = self.ddgs_client.text(
                    query,
                    max_results=num_results,
                    backend=backend,
                    region=self.config.ddgs_region,
                    safesearch=self.config.ddgs_safesearch,
                )

                if raw_results:
                    results = [
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", r.get("url", "")),
                            content=r.get("body", r.get("description", "")),
                        )
                        for r in raw_results
                    ]
                    logger.info(
                        f"DDGS search succeeded with {backend}: {len(results)} results"
                    )
                    return results

            except Exception as e:
                last_error = e
                logger.warning(f"DDGS {backend} failed: {e}")
                continue

        if last_error:
            logger.error(f"All DDGS backends failed. Last error: {last_error}")

        return []

    def _format_results(
        self,
        results: list[SearchResult],
        answer: Optional[str] = None,
        include_raw_content: bool = False,
    ) -> str:
        """Format search results for display."""
        output_parts = []

        # Include AI-generated answer if available
        if answer:
            output_parts.append(f"## AI Summary\n{answer}\n")

        output_parts.append("## Search Results")

        for i, r in enumerate(results, 1):
            result_text = (
                f"### {i}. {r.title}\n"
                f"**URL:** {r.url}\n"
                f"**Excerpt:** {r.content or 'N/A'}"
            )
            if r.score is not None:
                result_text += f"\n**Relevance:** {r.score:.2f}"
            if include_raw_content and r.raw_content:
                # Truncate raw content to avoid overwhelming output
                content_preview = r.raw_content[:2000]
                if len(r.raw_content) > 2000:
                    content_preview += "... [truncated]"
                result_text += f"\n**Full Content:**\n{content_preview}"

            output_parts.append(result_text)

        return "\n\n".join(output_parts)

    @command(
        ["web_search", "search"],
        "Search the web for information. Uses the best available search provider.",
        {
            "query": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The search query",
                required=True,
            ),
            "num_results": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Number of results to return (1-20)",
                minimum=1,
                maximum=20,
                required=False,
            ),
        },
    )
    def web_search(self, query: str, num_results: int = 8) -> str:
        """
        Search the web using the best available provider.

        Automatically selects provider: Tavily > Serper > DDGS (multi-engine)

        Args:
            query: The search query
            num_results: Number of results to return (default: 8)

        Returns:
            Formatted search results with optional AI summary
        """
        provider = self._get_provider()
        results: list[SearchResult] = []
        answer: Optional[str] = None

        # Try primary provider
        try:
            if provider == SearchProvider.TAVILY:
                results, answer = self._search_tavily(
                    query,
                    num_results,
                    include_answer=self.config.tavily_include_answer,
                )
            elif provider == SearchProvider.SERPER:
                results = self._search_serper(query, num_results)
            else:
                results = self._search_ddgs(query, num_results)

        except Exception as e:
            logger.warning(f"{provider.value} search failed: {e}, trying fallback...")

            # Fallback chain
            if provider == SearchProvider.TAVILY and self.config.serper_api_key:
                try:
                    results = self._search_serper(query, num_results)
                    provider = SearchProvider.SERPER
                except Exception as e2:
                    logger.warning(f"Serper fallback failed: {e2}")

            if not results:
                logger.info("Falling back to DDGS multi-engine search")
                results = self._search_ddgs(query, num_results)
                provider = SearchProvider.DDGS

        if not results:
            return "No search results found."

        logger.info(f"Search completed using {provider.value}: {len(results)} results")
        return self._format_results(results, answer)

    @command(
        ["search_and_extract"],
        "Search and extract full content from web pages. Best for research tasks.",
        {
            "query": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The search query",
                required=True,
            ),
            "num_results": JSONSchema(
                type=JSONSchema.Type.INTEGER,
                description="Number of results to return (1-10)",
                minimum=1,
                maximum=10,
                required=False,
            ),
        },
    )
    def search_and_extract(self, query: str, num_results: int = 5) -> str:
        """
        Search and extract full page content using Tavily's advanced search.

        This command performs a deep search and extracts the full content
        from the most relevant pages. Best for research tasks that need
        comprehensive information.

        Args:
            query: The search query
            num_results: Number of results with full content (default: 5)

        Returns:
            Search results with extracted page content
        """
        if not self.config.tavily_api_key:
            return (
                "Error: search_and_extract requires a Tavily API key. "
                "Set TAVILY_API_KEY environment variable."
            )

        try:
            results, answer = self._search_tavily(
                query,
                num_results,
                include_answer=True,
                include_raw_content=True,
                search_depth="advanced",
            )

            if not results:
                return "No search results found."

            return self._format_results(results, answer, include_raw_content=True)

        except Exception as e:
            logger.error(f"search_and_extract failed: {e}")
            return f"Search failed: {e}"

    # Legacy method for backwards compatibility
    def safe_google_results(self, results: str | list) -> str:
        """Return the results of a Google search in a safe format."""
        if isinstance(results, list):
            safe_message = json.dumps(
                [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
            )
        else:
            safe_message = results.encode("utf-8", "ignore").decode("utf-8")
        return safe_message
