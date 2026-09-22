from typing import Any

from tavily import AsyncTavilyClient

from backend.data.model import NodeExecutionStats
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._api import (
    CREDIT_USD,
    TavilySearchDepth,
    TavilySearchResult,
    TavilyTimeRange,
    TavilyTopic,
    credits_from_response,
)
from ._config import tavily


class TavilySearchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = tavily.credentials_field(
            description="The Tavily integration requires an API Key."
        )
        query: str = SchemaField(description="The search query")
        topic: TavilyTopic = SchemaField(
            description="Search category: general, news, or finance",
            default=TavilyTopic.GENERAL,
            advanced=True,
        )
        search_depth: TavilySearchDepth = SchemaField(
            description="Depth of the search: basic, fast or ultra-fast (1 credit), or advanced (2 credits)",
            default=TavilySearchDepth.BASIC,
            advanced=True,
        )
        max_results: int = SchemaField(
            description="Maximum number of results to return",
            default=5,
            ge=1,
            le=20,
            advanced=True,
        )
        time_range: TavilyTimeRange | None = SchemaField(
            description="Only include results published within this time range",
            default=None,
            advanced=True,
        )
        include_domains: list[str] = SchemaField(
            description="Domains to include in search", default_factory=list
        )
        exclude_domains: list[str] = SchemaField(
            description="Domains to exclude from search",
            default_factory=list,
            advanced=True,
        )
        include_answer: bool = SchemaField(
            description="Include an LLM-generated answer to the query, based on the search results",
            default=True,
        )
        include_raw_content: bool = SchemaField(
            description="Include the full page content for each result",
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[TavilySearchResult] = SchemaField(
            description="List of search results"
        )
        result: TavilySearchResult = SchemaField(description="Single search result")
        answer: str = SchemaField(
            description="LLM-generated answer to the query, based on the search results"
        )
        context: str = SchemaField(
            description="A formatted string of the search results ready for LLMs."
        )
        error: str = SchemaField(
            description="Error message if the search failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="363bb641-2147-47a0-9aaf-caae26b06dcb",
            description="Searches the web using Tavily's AI-native search API",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": tavily.get_test_credentials().model_dump(),
                "query": "What is AutoGPT?",
                "max_results": 1,
            },
            test_credentials=tavily.get_test_credentials(),
            test_output=[
                ("results", lambda x: isinstance(x, list) and len(x) == 1),
                ("result", lambda x: x.url == "https://agpt.co"),
                ("answer", "AutoGPT is a platform for building AI agents."),
                ("context", lambda x: "AutoGPT" in x),
            ],
            test_mock={
                "_search": lambda *args, **kwargs: {
                    "query": "What is AutoGPT?",
                    "answer": "AutoGPT is a platform for building AI agents.",
                    "results": [
                        {
                            "title": "AutoGPT",
                            "url": "https://agpt.co",
                            "content": "AutoGPT is a platform for building AI agents.",
                            "score": 0.98,
                        }
                    ],
                    "usage": {"credits": 1},
                }
            },
        )

    async def _search(self, credentials: APIKeyCredentials, **kwargs) -> dict[str, Any]:
        """Private method to call the Tavily API - can be mocked for testing."""
        client = AsyncTavilyClient(api_key=credentials.api_key.get_secret_value())
        return await client.search(**kwargs)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        sdk_kwargs: dict[str, Any] = {
            "query": input_data.query,
            "topic": input_data.topic.value,
            "search_depth": input_data.search_depth.value,
            "max_results": input_data.max_results,
            "include_answer": input_data.include_answer,
            "include_raw_content": input_data.include_raw_content,
            "include_usage": True,
        }
        if input_data.time_range:
            sdk_kwargs["time_range"] = input_data.time_range.value
        if input_data.include_domains:
            sdk_kwargs["include_domains"] = input_data.include_domains
        if input_data.exclude_domains:
            sdk_kwargs["exclude_domains"] = input_data.exclude_domains

        try:
            response = await self._search(credentials, **sdk_kwargs)
        except Exception as e:
            raise BlockExecutionError(
                message=f"Search failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        # Actual credit spend from the API's usage report; falls back to the
        # documented schedule (basic search 1 credit, advanced 2).
        credits_used = credits_from_response(
            response,
            estimate=2 if input_data.search_depth == TavilySearchDepth.ADVANCED else 1,
        )
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=credits_used * CREDIT_USD,
                provider_cost_type="cost_usd",
            )
        )

        results = [
            TavilySearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                raw_content=r.get("raw_content"),
            )
            for r in response.get("results", [])
        ]

        yield "results", results
        for result in results:
            yield "result", result

        if response.get("answer"):
            yield "answer", response["answer"]

        yield "context", "\n\n".join(
            f"[{r.title}]({r.url})\n{r.content}" for r in results
        )
