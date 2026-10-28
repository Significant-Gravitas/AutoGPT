from typing import Any

from pydantic import model_validator

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    NodeExecutionStats,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._api import (
    MAX_DOMAIN_FILTERS,
    PerplexityRecencyFilter,
    PerplexitySearchResponse,
    PerplexitySearchResult,
    PerplexitySearchType,
    check_domain_filter,
    search,
    search_cost_usd,
)
from ._config import perplexity


class PerplexitySearchBlock(Block):
    """Search the web with the Perplexity Search API."""

    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = perplexity.credentials_field(
            description="Perplexity API credentials"
        )
        query: str = SchemaField(description="The search query")
        max_results: int = SchemaField(
            description="Maximum number of results to return (1 to 20)",
            default=5,
            ge=1,
            le=20,
        )
        search_type: PerplexitySearchType = SchemaField(
            description=(
                "Web search returns the highest quality results; "
                "fast search returns results sooner"
            ),
            default=PerplexitySearchType.WEB,
            advanced=True,
        )
        search_domain_filter: list[str] = SchemaField(
            description=(
                "Up to 20 domains without protocol, e.g. nature.com. Either list "
                "only domains to include, or prefix every entry with '-' to "
                "exclude those domains; one request cannot mix the two."
            ),
            default_factory=list,
            max_length=MAX_DOMAIN_FILTERS,
            advanced=True,
        )
        search_recency_filter: PerplexityRecencyFilter | None = SchemaField(
            description="Only return pages published within this time window",
            default=None,
            advanced=True,
        )

        @model_validator(mode="after")
        def _check_domain_filter(self):
            check_domain_filter(self.search_domain_filter)
            return self

    class Output(BlockSchemaOutput):
        results: list[PerplexitySearchResult] = SchemaField(
            description="List of search results"
        )
        result: PerplexitySearchResult = SchemaField(
            description="Each search result, emitted one at a time"
        )
        context: str = SchemaField(
            description="All results formatted as text for use in a prompt"
        )

    def __init__(self):
        super().__init__(
            id="aed4fa4a-793a-4420-b43e-385c673d52a2",
            description="Search the web with the Perplexity Search API",
            categories={BlockCategory.SEARCH},
            effect=BlockEffect.READ,
            input_schema=PerplexitySearchBlock.Input,
            output_schema=PerplexitySearchBlock.Output,
            test_input={
                "query": "AutoGPT",
                "credentials": perplexity.get_test_credentials().model_dump(),
            },
            test_credentials=perplexity.get_test_credentials(),
            test_output=[
                ("results", lambda x: len(x) == 1),
                (
                    "result",
                    lambda x: x.url
                    == "https://github.com/Significant-Gravitas/AutoGPT",
                ),
                ("context", lambda x: "AutoGPT" in x),
            ],
            test_mock={
                "_search": lambda *args, **kwargs: PerplexitySearchResponse(
                    results=[
                        PerplexitySearchResult(
                            title="AutoGPT",
                            url="https://github.com/Significant-Gravitas/AutoGPT",
                            snippet="AutoGPT is a platform for AI agents.",
                            date="2024-01-01",
                        )
                    ]
                ),
            },
        )

    async def _search(
        self, credentials: APIKeyCredentials, payload: dict[str, Any]
    ) -> PerplexitySearchResponse:
        return await search(credentials, payload)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload: dict[str, Any] = {
            "query": input_data.query,
            "max_results": input_data.max_results,
            "search_type": input_data.search_type.value,
        }
        if input_data.search_domain_filter:
            payload["search_domain_filter"] = input_data.search_domain_filter
        if input_data.search_recency_filter:
            payload["search_recency_filter"] = input_data.search_recency_filter.value

        try:
            response = await self._search(credentials, payload)
        except ValueError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e

        # A successful request is billed even when it returns no results.
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=search_cost_usd(input_data.search_type),
                provider_cost_type="cost_usd",
            )
        )

        results = response.results
        yield "results", results
        for result in results:
            yield "result", result
        yield "context", "\n\n".join(
            f"Title: {r.title}\nURL: {r.url}\nSnippet: {r.snippet}" for r in results
        )
