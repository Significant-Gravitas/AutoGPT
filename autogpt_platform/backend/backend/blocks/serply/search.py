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
from backend.util.exceptions import BlockExecutionError

from ._api import (
    MAX_RESULTS,
    SEARCH_COST_USD,
    TIME_RANGE_TBS,
    SerplyClient,
    SerplySearchResult,
    SerplyTimeRange,
    results_from_response,
)
from ._config import serply


class SerplySearchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = serply.credentials_field(
            description="The Serply integration requires an API Key."
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
        time_range: SerplyTimeRange | None = SchemaField(
            description="Only include results published within this time range",
            default=None,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[SerplySearchResult] = SchemaField(
            description="List of search results"
        )
        result: SerplySearchResult = SchemaField(description="Single search result")
        context: str = SchemaField(
            description="A formatted string of the search results ready for LLMs."
        )
        error: str = SchemaField(
            description="Error message if the search failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="5b0f0c1e-7f62-4a53-9a1d-3d2d6c5e8a41",
            description="Searches Google using Serply's search API",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            test_input={
                "credentials": serply.get_test_credentials().model_dump(),
                "query": "What is AutoGPT?",
                "max_results": 1,
            },
            test_credentials=serply.get_test_credentials(),
            test_output=[
                ("results", lambda x: isinstance(x, list) and len(x) == 1),
                ("result", lambda x: x.url == "https://agpt.co"),
                ("context", lambda x: "AutoGPT" in x),
            ],
            test_mock={
                "_search": lambda *args, **kwargs: {
                    "results": [
                        {
                            "title": "AutoGPT",
                            "link": "https://agpt.co",
                            "description": "AutoGPT is a platform for building AI agents.",
                            "position": 1,
                        }
                    ],
                }
            },
            effect=BlockEffect.READ,
        )

    async def _search(
        self, credentials: APIKeyCredentials, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Private method to call the Serply API - can be mocked for testing."""
        return await SerplyClient(credentials).search(params)

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params: dict[str, Any] = {
            "q": input_data.query,
            "num": MAX_RESULTS,
        }
        if input_data.country is not None:
            params["gl"] = input_data.country
        if input_data.time_range is not None:
            params["tbs"] = TIME_RANGE_TBS[input_data.time_range]

        try:
            response = await self._search(credentials, params)
            # num is only an approximate bound, so trim to the caller's limit here.
            results = results_from_response(response)[: input_data.max_results]
        except Exception as e:
            raise BlockExecutionError(
                message=f"Search failed: {e}",
                block_name=self.name,
                block_id=self.id,
            ) from e

        self.merge_stats(
            NodeExecutionStats(
                provider_cost=SEARCH_COST_USD,
                provider_cost_type="cost_usd",
            )
        )

        yield "results", results
        for result in results:
            yield "result", result

        yield "context", "\n\n".join(
            f"[{r.title}]({r.url})\n{r.description}" for r in results
        )
