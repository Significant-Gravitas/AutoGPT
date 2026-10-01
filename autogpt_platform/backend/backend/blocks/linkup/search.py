from datetime import date

from linkup import (
    LinkupClient,
    LinkupSearchResults,
    LinkupSearchTextResult,
    LinkupSource,
    LinkupSourcedAnswer,
)

from backend.data.model import NodeExecutionStats
from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)
from backend.util.exceptions import BlockExecutionError

from ._api import (
    LinkupAnswerSource,
    LinkupOutputType,
    LinkupSearchDepth,
    LinkupSearchParams,
    LinkupSearchResult,
    search_cost_usd,
)
from ._config import linkup


class LinkupSearchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = linkup.credentials_field(
            description="The Linkup integration requires an API Key."
        )
        query: str = SchemaField(description="The search query")
        output_type: LinkupOutputType = SchemaField(
            description="searchResults returns a list of relevant sources; sourcedAnswer returns an LLM-generated answer with the sources supporting it",
            default="searchResults",
        )
        depth: LinkupSearchDepth = SchemaField(
            description="Depth of the search: fast or standard for most queries, deep for complex multi-step questions (slower, 10x the cost)",
            default="standard",
            advanced=True,
        )
        max_results: int = SchemaField(
            description="Maximum number of results to return",
            default=10,
            ge=1,
            advanced=True,
        )
        include_domains: list[str] = SchemaField(
            description="Domains to restrict the search to", default_factory=list
        )
        exclude_domains: list[str] = SchemaField(
            description="Domains to exclude from search",
            default_factory=list,
            advanced=True,
        )
        from_date: date | None = SchemaField(
            description="Only include sources published on or after this date",
            default=None,
            advanced=True,
        )
        to_date: date | None = SchemaField(
            description="Only include sources published on or before this date",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[LinkupSearchResult] = SchemaField(
            description="List of search results (searchResults output type)"
        )
        result: LinkupSearchResult = SchemaField(description="Single search result")
        context: str = SchemaField(
            description="A formatted string of the search results ready for LLMs."
        )
        answer: str = SchemaField(
            description="LLM-generated answer to the query (sourcedAnswer output type)"
        )
        sources: list[LinkupAnswerSource] = SchemaField(
            description="Sources supporting the answer (sourcedAnswer output type)"
        )
        error: str = SchemaField(
            description="Error message if the search failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="2c613750-8788-447d-9d62-a2998ed1e52d",
            description="Searches the web in real time using Linkup, returning relevant sources or a sourced answer",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input=[
                {
                    "credentials": linkup.get_test_credentials().model_dump(),
                    "query": "What is AutoGPT?",
                    "max_results": 1,
                },
                {
                    "credentials": linkup.get_test_credentials().model_dump(),
                    "query": "What is AutoGPT?",
                    "output_type": "sourcedAnswer",
                },
            ],
            test_credentials=linkup.get_test_credentials(),
            test_output=[
                ("results", lambda x: isinstance(x, list) and len(x) == 1),
                ("result", lambda x: x.url == "https://agpt.co"),
                ("context", lambda x: "AutoGPT" in x),
                ("answer", "AutoGPT is a platform for building AI agents."),
                ("sources", lambda x: x[0].url == "https://agpt.co"),
            ],
            test_mock={
                "_search_results": lambda *args, **kwargs: LinkupSearchResults(
                    results=[
                        LinkupSearchTextResult(
                            type="text",
                            name="AutoGPT",
                            url="https://agpt.co",
                            content="AutoGPT is a platform for building AI agents.",
                            favicon="",
                        )
                    ]
                ),
                "_sourced_answer": lambda *args, **kwargs: LinkupSourcedAnswer(
                    answer="AutoGPT is a platform for building AI agents.",
                    sources=[
                        LinkupSource(
                            name="AutoGPT",
                            url="https://agpt.co",
                            snippet="AutoGPT is a platform for building AI agents.",
                            favicon="",
                        )
                    ],
                ),
            },
            effect=BlockEffect.READ,
        )

    async def _search_results(
        self, credentials: APIKeyCredentials, params: LinkupSearchParams
    ) -> LinkupSearchResults:
        """Private method to call the Linkup API - can be mocked for testing."""
        client = LinkupClient(api_key=credentials.api_key)
        return await client.async_search(output_type="searchResults", **params)

    async def _sourced_answer(
        self, credentials: APIKeyCredentials, params: LinkupSearchParams
    ) -> LinkupSourcedAnswer:
        """Private method to call the Linkup API - can be mocked for testing."""
        client = LinkupClient(api_key=credentials.api_key)
        return await client.async_search(output_type="sourcedAnswer", **params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params = LinkupSearchParams(
            query=input_data.query,
            depth=input_data.depth,
            max_results=input_data.max_results,
            include_domains=input_data.include_domains or None,
            exclude_domains=input_data.exclude_domains or None,
            from_date=input_data.from_date,
            to_date=input_data.to_date,
        )
        cost_usd = search_cost_usd(input_data.depth, input_data.output_type)

        if input_data.output_type == "sourcedAnswer":
            outputs = self._run_sourced_answer(credentials, params, cost_usd)
        else:
            outputs = self._run_search_results(credentials, params, cost_usd)

        async for name, value in outputs:
            yield name, value

    async def _run_search_results(
        self,
        credentials: APIKeyCredentials,
        params: LinkupSearchParams,
        cost_usd: float,
    ) -> BlockOutput:
        try:
            response = await self._search_results(credentials, params)
        except Exception as e:
            raise self._search_error(e) from e
        self._merge_cost(cost_usd)

        results = [
            LinkupSearchResult(name=r.name, url=r.url, content=r.content)
            for r in response.results
            if r.type == "text"
        ]

        yield "results", results
        for result in results:
            yield "result", result

        yield "context", "\n\n".join(
            f"[{r.name}]({r.url})\n{r.content}" for r in results
        )

    async def _run_sourced_answer(
        self,
        credentials: APIKeyCredentials,
        params: LinkupSearchParams,
        cost_usd: float,
    ) -> BlockOutput:
        try:
            response = await self._sourced_answer(credentials, params)
        except Exception as e:
            raise self._search_error(e) from e
        self._merge_cost(cost_usd)

        yield "answer", response.answer
        yield "sources", [
            LinkupAnswerSource(name=s.name, url=s.url, snippet=s.snippet)
            for s in response.sources
        ]

    def _merge_cost(self, cost_usd: float) -> None:
        self.merge_stats(
            NodeExecutionStats(provider_cost=cost_usd, provider_cost_type="cost_usd")
        )

    def _search_error(self, e: Exception) -> BlockExecutionError:
        return BlockExecutionError(
            message=f"Search failed: {e}",
            block_name=self.name,
            block_id=self.id,
        )
