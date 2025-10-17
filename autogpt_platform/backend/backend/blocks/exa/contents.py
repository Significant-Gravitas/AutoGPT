from enum import Enum
from typing import Optional

from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
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
    class Input(BlockSchema):
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

    class Output(BlockSchema):
        results: list[ExaSearchResults] = SchemaField(
            description="List of document contents with metadata", default_factory=list
        )
        result: ExaSearchResults = SchemaField(
            description="Single document content result"
        )
        context: str = SchemaField(
            description="A formatted string of the results ready for LLMs", default=""
        )
        request_id: str = SchemaField(
            description="Unique identifier for the request", default=""
        )
        statuses: list[ContentStatus] = SchemaField(
            description="Status information for each requested URL",
            default_factory=list,
        )
        cost_dollars: Optional[CostDollars] = SchemaField(
            description="Cost breakdown for the request", default=None
        )
        error: str = SchemaField(
            description="Error message if the request failed", default=""
        )

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
        url = "https://api.exa.ai/contents"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        # Build payload with urls or deprecated ids
        payload = {}

        # Prefer urls over ids
        if input_data.urls:
            payload["urls"] = input_data.urls
        elif input_data.ids:
            payload["ids"] = input_data.ids
        else:
            yield "error", "Either 'urls' or 'ids' must be provided"
            return

        # Handle text field - when true, include HTML tags for better LLM understanding
        if input_data.text:
            payload["text"] = {"includeHtmlTags": True}

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
            payload["highlights"] = highlights_dict

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
            payload["summary"] = summary_dict

        # Handle livecrawl
        if input_data.livecrawl:
            payload["livecrawl"] = input_data.livecrawl.value

        # Handle livecrawl_timeout
        if input_data.livecrawl_timeout is not None:
            payload["livecrawlTimeout"] = input_data.livecrawl_timeout

        # Handle subpages
        if input_data.subpages is not None:
            payload["subpages"] = input_data.subpages

        # Handle subpage_target
        if input_data.subpage_target:
            payload["subpageTarget"] = input_data.subpage_target

        # Handle extras - only include if modified from defaults
        if input_data.extras and (
            input_data.extras.links > 0 or input_data.extras.image_links > 0
        ):
            extras_dict = {}
            if input_data.extras.links:
                extras_dict["links"] = input_data.extras.links
            if input_data.extras.image_links:
                extras_dict["imageLinks"] = input_data.extras.image_links
            payload["extras"] = extras_dict

        # Always enable context for LLM-ready output
        payload["context"] = True

        try:
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            # Extract all response fields
            yield "results", data.get("results", [])

            # Yield individual results
            for result in data.get("results", []):
                yield "result", result

            # Yield context if present
            if "context" in data:
                yield "context", data["context"]

            # Yield request ID if present
            if "requestId" in data:
                yield "request_id", data["requestId"]

            # Yield statuses if present
            if "statuses" in data:
                yield "statuses", data["statuses"]

            # Yield cost information if present
            if "costDollars" in data:
                yield "cost_dollars", data["costDollars"]

        except Exception as e:
            yield "error", str(e)
