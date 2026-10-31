from datetime import datetime
from typing import Any

from pydantic import BaseModel

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

from ._config import exa
from ._exa_rest import EXA_API_BASE, ExaRestMixin
from ._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from .helpers import (
    ContentSettings,
    CostDollars,
    ExaSearchResults,
    merge_exa_cost,
    parse_cost_dollars,
    process_contents_settings,
)
from .search_options import (
    CATEGORY_DESCRIPTION,
    LEGACY_DEFAULT_CONTENTS,
    ExaSearchCategories,
    category_value,
    iso_or_none,
)


class ExaFindSimilarResponse(BaseModel):
    requestId: str = ""
    results: list[ExaSearchResults] = []
    context: str | None = None


class ExaFindSimilarBlock(ExaRestMixin, Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        url: str = SchemaField(
            description="The url for which you would like to find similar links"
        )
        number_of_results: int = SchemaField(
            description="Number of results to return (1 to 100)",
            default=10,
            ge=1,
            le=100,
            advanced=True,
        )
        exclude_source_domain: bool = SchemaField(
            description="Leave out results from the same domain as the source URL",
            default=False,
            advanced=True,
        )
        category: ExaSearchCategories | None = SchemaField(
            title="Category",
            description=CATEGORY_DESCRIPTION,
            default=None,
            advanced=True,
        )
        include_domains: list[str] = SchemaField(
            description="List of domains to include in the search. If specified, results will only come from these domains.",
            default_factory=list,
            advanced=True,
        )
        exclude_domains: list[str] = SchemaField(
            description="Domains to exclude from search",
            default_factory=list,
            advanced=True,
        )
        start_crawl_date: datetime | None = SchemaField(
            description="Deprecated by Exa and ignored by the API, so this block "
            "no longer sends it. Use Start Published Date instead.",
            advanced=True,
            default=None,
        )
        end_crawl_date: datetime | None = SchemaField(
            description="Deprecated by Exa and ignored by the API, so this block "
            "no longer sends it. Use End Published Date instead.",
            advanced=True,
            default=None,
        )
        start_published_date: datetime | None = SchemaField(
            description="Start date for published content", advanced=True, default=None
        )
        end_published_date: datetime | None = SchemaField(
            description="End date for published content", advanced=True, default=None
        )
        include_text: list[str] = SchemaField(
            description="Deprecated by Exa. Strings that must appear in the page "
            "text (approximate, word-level matching)",
            default_factory=list,
            advanced=True,
        )
        exclude_text: list[str] = SchemaField(
            description="Deprecated by Exa. Strings that must not appear in the "
            "page text",
            default_factory=list,
            advanced=True,
        )
        contents: ContentSettings = SchemaField(
            description="What to return for each result. Left empty, each result "
            "gets up to 10,000 characters of page text.",
            default=ContentSettings(),
            advanced=True,
        )
        moderation: bool = SchemaField(
            description="Legacy: Exa's Find Similar API no longer documents "
            "moderation; still sent when enabled.",
            default=False,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[ExaSearchResults] = SchemaField(
            description="List of similar documents with metadata and content"
        )
        result: ExaSearchResults = SchemaField(
            description="Single similar document result"
        )
        context: str = SchemaField(
            description="Deprecated by Exa: the results as one string for LLMs, "
            "returned only when Contents > Context is enabled."
        )
        request_id: str = SchemaField(description="Unique identifier for the request")
        cost_dollars: CostDollars | None = SchemaField(
            description="Cost breakdown for the request"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="5e7315d1-af61-4a0c-9350-7c868fa7438a",
            description="Deprecated by Exa: prefer the Exa Search block with a "
            "query describing the source page. Finds pages similar to a URL "
            "using Exa's findSimilar API.",
            categories={BlockCategory.SEARCH},
            input_schema=ExaFindSimilarBlock.Input,
            output_schema=ExaFindSimilarBlock.Output,
            effect=BlockEffect.READ,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "url": "https://example.com",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                (
                    "results",
                    [ExaSearchResults(id="s1", url="https://similar.com", score=0.9)],
                ),
                (
                    "result",
                    ExaSearchResults(id="s1", url="https://similar.com", score=0.9),
                ),
                ("request_id", "req-1"),
                ("cost_dollars", CostDollars(total=0.005)),
            ],
            test_mock={
                "call_exa": lambda *args, **kwargs: {
                    "requestId": "req-1",
                    "results": [
                        {"id": "s1", "url": "https://similar.com", "score": 0.9}
                    ],
                    "costDollars": {"total": 0.005},
                }
            },
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        payload = build_find_similar_payload(input_data)
        data = await self.call_exa(
            credentials, "POST", f"{EXA_API_BASE}/findSimilar", body=payload
        )
        response = ExaFindSimilarResponse.model_validate(data)

        yield "results", response.results
        for result in response.results:
            yield "result", result
        if response.context:
            yield "context", response.context
        if response.requestId:
            yield "request_id", response.requestId
        cost = parse_cost_dollars(data)
        if cost is not None:
            yield "cost_dollars", cost
        merge_exa_cost(self, data)


def build_find_similar_payload(
    input_data: ExaFindSimilarBlock.Input,
) -> dict[str, Any]:
    """Build the ``/findSimilar`` body; unset options and the ignored crawl
    dates are omitted."""
    optional: dict[str, Any] = {
        "excludeSourceDomain": input_data.exclude_source_domain or None,
        "category": category_value(input_data.category),
        "includeDomains": input_data.include_domains,
        "excludeDomains": input_data.exclude_domains,
        "startPublishedDate": iso_or_none(input_data.start_published_date),
        "endPublishedDate": iso_or_none(input_data.end_published_date),
        "includeText": input_data.include_text,
        "excludeText": input_data.exclude_text,
        "moderation": input_data.moderation or None,
    }
    return {
        "url": input_data.url,
        "numResults": input_data.number_of_results,
        "contents": process_contents_settings(input_data.contents)
        or LEGACY_DEFAULT_CONTENTS,
        **{key: value for key, value in optional.items() if value},
    }
