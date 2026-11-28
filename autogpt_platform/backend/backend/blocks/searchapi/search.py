from typing import Any

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

from ._api import (
    MAX_RESULTS,
    SEARCH_COST_USD,
    TIME_RANGE_PARAM,
    SearchApiClient,
    SearchApiSearchResult,
    SearchApiTimeRange,
    results_from_response,
)
from ._config import searchapi


class SearchApiSearchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = searchapi.credentials_field(
            description="The SearchApi integration requires an API Key."
        )
        query: str = SchemaField(
            description="The search query; supports Google operators such as site: and filetype:"
        )
        max_results: int = SchemaField(
            description="Maximum number of results to return",
            default=5,
            ge=1,
            le=MAX_RESULTS,
            advanced=True,
        )
        country: str | None = SchemaField(
            description="Two-letter country code to localize results (e.g. us, gb, jp)",
            default=None,
            advanced=True,
        )
        time_range: SearchApiTimeRange | None = SchemaField(
            description="Only include results published within this time range",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[SearchApiSearchResult] = SchemaField(
            description="List of search results"
        )
        result: SearchApiSearchResult = SchemaField(description="Single search result")
        context: str = SchemaField(
            description="A formatted string of the search results ready for LLMs."
        )
        error: str = SchemaField(
            description="Error message if the search failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="cc719fe2-008d-47c6-9cfc-e849d810f9bc",
            description="Searches Google using SearchApi's search API",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": searchapi.get_test_credentials().model_dump(),
                "query": "What is AutoGPT?",
                "max_results": 1,
            },
            test_credentials=searchapi.get_test_credentials(),
            test_output=[
                ("results", lambda x: isinstance(x, list) and len(x) == 1),
                ("result", lambda x: x.url == "https://agpt.co"),
                ("context", lambda x: "AutoGPT" in x),
            ],
            test_mock={
                "_search": lambda *args, **kwargs: {
                    "organic_results": [
                        {
                            "position": 1,
                            "title": "AutoGPT",
                            "link": "https://agpt.co",
                            "snippet": "AutoGPT is a platform for building AI agents.",
                        }
                    ],
                }
            },
            effect=BlockEffect.READ,
        )

    async def _search(
        self, credentials: APIKeyCredentials, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Private method to call the SearchApi API - can be mocked for testing."""
        return await SearchApiClient(credentials).search(params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params: dict[str, Any] = {
            "engine": "google",
            "q": input_data.query,
            "num": input_data.max_results,
        }
        if input_data.country is not None:
            params["gl"] = input_data.country
        if input_data.time_range is not None:
            params["time_period"] = TIME_RANGE_PARAM[input_data.time_range]

        # An HTTP failure raises ValueError from the client, which Block.execute
        # turns into BlockExecutionError; anything unexpected propagates as is.
        response = await self._search(credentials, params)

        # SearchApi bills a request once it answers 2xx, whatever the body
        # holds, so the cost is recorded before the body is read.
        self.merge_stats(
            NodeExecutionStats(
                provider_cost=SEARCH_COST_USD,
                provider_cost_type="cost_usd",
            )
        )

        # num is honoured up to 10 but is a bound, not a promise, so trim
        # to the caller's limit here.
        results = results_from_response(response)[: input_data.max_results]

        yield "results", results
        for result in results:
            yield "result", result

        yield "context", "\n\n".join(
            f"[{r.title}]({r.url})\n{r.description}" for r in results
        )
