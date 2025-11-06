from enum import Enum
from typing import Optional

from exa_py import AsyncExa
from pydantic import BaseModel

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

from ._config import exa
from .helpers import (
    CostDollars,
    ExaSearchResults,
    ExtrasSettings,
    HighlightSettings,
    LivecrawlTypes,
    SummarySettings,
)


class ContentStatusTag(str, Enum):
    CRAWL_NOT_FOUND = "CRAWL_NOT_FOUND"
    CRAWL_TIMEOUT = "CRAWL_TIMEOUT"
    CRAWL_LIVECRAWL_TIMEOUT = "CRAWL_LIVECRAWL_TIMEOUT"
    SOURCE_NOT_AVAILABLE = "SOURCE_NOT_AVAILABLE"
    CRAWL_UNKNOWN_ERROR = "CRAWL_UNKNOWN_ERROR"


class ContentError(BaseModel):
    tag: Optional[ContentStatusTag] = SchemaField(
        default=None, description="Specific error type"
    )
    httpStatusCode: Optional[int] = SchemaField(
        default=None, description="The corresponding HTTP status code"
    )


class ContentStatus(BaseModel):
    id: str = SchemaField(description="The URL that was requested")
    status: str = SchemaField(
        description="Status of the content fetch operation (success or error)"
    )
    error: Optional[ContentError] = SchemaField(
        default=None, description="Error details, only present when status is 'error'"
    )


class ExaContentsBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        urls: list[str] = SchemaField(
            description="Array of URLs to crawl (preferred over 'ids')",
            default_factory=list,
            advanced=False,
        )
        ids: list[str] = SchemaField(
            description="[DEPRECATED - use 'urls' instead] Array of document IDs obtained from searches",
            default_factory=list,
            advanced=True,
        )
        text: bool = SchemaField(
            description="Retrieve text content from pages",
            default=True,
        )
        highlights: HighlightSettings = SchemaField(
            description="Text snippets most relevant from each page",
            default=HighlightSettings(),
        )
        summary: SummarySettings = SchemaField(
            description="LLM-generated summary of the webpage",
            default=SummarySettings(),
        )
        livecrawl: Optional[LivecrawlTypes] = SchemaField(
            description="Livecrawling options: never, fallback (default), always, preferred",
            default=LivecrawlTypes.FALLBACK,
            advanced=True,
        )
        livecrawl_timeout: Optional[int] = SchemaField(
            description="Timeout for livecrawling in milliseconds",
            default=10000,
            advanced=True,
        )
        subpages: Optional[int] = SchemaField(
            description="Number of subpages to crawl", default=0, ge=0, advanced=True
        )
        subpage_target: Optional[str | list[str]] = SchemaField(
            description="Keyword(s) to find specific subpages of search results",
            default=None,
            advanced=True,
        )
        extras: ExtrasSettings = SchemaField(
            description="Extra parameters for additional content",
            default=ExtrasSettings(),
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        results: list[ExaSearchResults] = SchemaField(
            description="List of document contents with metadata"
        )
        result: ExaSearchResults = SchemaField(
            description="Single document content result"
        )
        context: str = SchemaField(
            description="A formatted string of the results ready for LLMs"
        )
        request_id: str = SchemaField(description="Unique identifier for the request")
        statuses: list[ContentStatus] = SchemaField(
            description="Status information for each requested URL"
        )
        cost_dollars: Optional[CostDollars] = SchemaField(
            description="Cost breakdown for the request"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="c52be83f-f8cd-4180-b243-af35f986b461",
            description="Retrieves document contents using Exa's contents API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaContentsBlock.Input,
            output_schema=ExaContentsBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        if not input_data.urls and not input_data.ids:
            raise ValueError("Either 'urls' or 'ids' must be provided")

        sdk_kwargs = {}

        # Prefer urls over ids
        if input_data.urls:
            sdk_kwargs["urls"] = input_data.urls
        elif input_data.ids:
            sdk_kwargs["ids"] = input_data.ids

        if input_data.text:
            sdk_kwargs["text"] = {"includeHtmlTags": True}

        # Handle highlights - only include if modified from defaults
        if input_data.highlights and (
            input_data.highlights.num_sentences != 1
            or input_data.highlights.highlights_per_url != 1
            or input_data.highlights.query is not None
        ):
            highlights_dict = {}
            highlights_dict["numSentences"] = input_data.highlights.num_sentences
            highlights_dict["highlightsPerUrl"] = (
                input_data.highlights.highlights_per_url
            )
            if input_data.highlights.query:
                highlights_dict["query"] = input_data.highlights.query
            sdk_kwargs["highlights"] = highlights_dict

        # Handle summary - only include if modified from defaults
        if input_data.summary and (
            input_data.summary.query is not None
            or input_data.summary.schema is not None
        ):
            summary_dict = {}
            if input_data.summary.query:
                summary_dict["query"] = input_data.summary.query
            if input_data.summary.schema:
                summary_dict["schema"] = input_data.summary.schema
            sdk_kwargs["summary"] = summary_dict

        if input_data.livecrawl:
            sdk_kwargs["livecrawl"] = input_data.livecrawl.value

        if input_data.livecrawl_timeout is not None:
            sdk_kwargs["livecrawl_timeout"] = input_data.livecrawl_timeout

        if input_data.subpages is not None:
            sdk_kwargs["subpages"] = input_data.subpages

        if input_data.subpage_target:
            sdk_kwargs["subpage_target"] = input_data.subpage_target

        # Handle extras - only include if modified from defaults
        if input_data.extras and (
            input_data.extras.links > 0 or input_data.extras.image_links > 0
        ):
            extras_dict = {}
            if input_data.extras.links:
                extras_dict["links"] = input_data.extras.links
            if input_data.extras.image_links:
                extras_dict["image_links"] = input_data.extras.image_links
            sdk_kwargs["extras"] = extras_dict

        # Always enable context for LLM-ready output
        sdk_kwargs["context"] = True

        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())
        response = await aexa.get_contents(**sdk_kwargs)

        converted_results = [
            ExaSearchResults.from_sdk(sdk_result)
            for sdk_result in response.results or []
        ]

        yield "results", converted_results

        for result in converted_results:
            yield "result", result

        if response.context:
            yield "context", response.context

        if response.statuses:
            yield "statuses", response.statuses

        if response.cost_dollars:
            yield "cost_dollars", response.cost_dollars
