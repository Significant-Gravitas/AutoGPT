from backend.sdk import (
    APIKeyCredentials,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
    CredentialsMetaInput,
    SchemaField,
)

from ._api import (
    MrScraperError,
    MrScraperSearchResult,
    error_message,
    json_object,
    web_search_request,
)
from ._base import MrScraperBlockBase
from ._config import mrscraper, mrscraper_credentials_field

_RESULT_FIELDS = set(MrScraperSearchResult.model_fields)


class MrScraperWebSearchBlock(MrScraperBlockBase):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = mrscraper_credentials_field()
        query: str = SchemaField(
            description="Search query text, trimmed before sending, containing 1 to "
            "1000 characters.",
            placeholder="site:example.com documentation",
        )
        country: str | None = SchemaField(
            description="Two-letter country code to search from, sent in lowercase.",
            default=None,
        )
        num_results: int = SchemaField(
            description="Number of search results to request, as an integer from 10 "
            "to 100. Fewer results may be returned.",
            default=10,
            ge=10,
            le=100,
        )

    class Output(BlockSchemaOutput):
        results: list[MrScraperSearchResult] = SchemaField(
            description="Ranked search results."
        )
        result: MrScraperSearchResult = SchemaField(
            description="One search result, emitted for each result in order."
        )
        token_usage: int = SchemaField(
            description="MrScraper tokens used by the search."
        )

    def __init__(self):
        super().__init__(
            id="9d8b3bc5-a368-465b-9f82-c6aae6dbbdb1",
            description="Search the web and return ranked results with titles, "
            "URLs, descriptions, and sources.",
            categories={BlockCategory.SEARCH},
            input_schema=self.Input,
            output_schema=self.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": mrscraper.get_test_credentials().model_dump(),
                "query": "site:example.com documentation",
                "country": "ID",
                "num_results": 20,
            },
            test_credentials=mrscraper.get_test_credentials(),
            test_output=[
                ("results", lambda r: [x.url for x in r] == ["https://example.com"]),
                ("result", lambda x: x.rank == 1 and x.source == "example.com"),
                ("token_usage", 1),
            ],
            test_mock={
                "_send": lambda *args, **kwargs: {
                    "success": True,
                    "message": "Example response",
                    "data": [
                        {
                            "rank": 1,
                            "title": "Example result",
                            "url": "https://example.com",
                            "description": "Example result description.",
                            "source": "example.com",
                        }
                    ],
                    "tokenUsage": 1,
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        request = web_search_request(
            query=input_data.query,
            country=input_data.country,
            num_results=input_data.num_results,
        )
        body = json_object(await self._send(credentials, request))
        items = body.get("data")
        if body.get("success") is False or not isinstance(items, list):
            raise MrScraperError(f"MrScraper search failed: {error_message(body)}")
        results = [
            MrScraperSearchResult(
                **{
                    k: v
                    for k, v in item.items()
                    if k in _RESULT_FIELDS and v is not None
                }
            )
            for item in items
            if isinstance(item, dict)
        ]
        yield "results", results
        for result in results:
            yield "result", result
        if isinstance(body.get("tokenUsage"), int):
            yield "token_usage", body["tokenUsage"]
