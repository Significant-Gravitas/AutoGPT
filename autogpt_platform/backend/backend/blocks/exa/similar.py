from datetime import datetime
from typing import Optional

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
    ContentSettings,
    CostDollars,
    ExaSearchResults,
    add_optional_fields,
    format_date_fields,
    process_contents_settings,
)


class ExaFindSimilarBlock(Block):
    class Input(BlockSchema):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        url: str = SchemaField(
            description="The url for which you would like to find similar links"
        )
        number_of_results: int = SchemaField(
            description="Number of results to return", default=10, advanced=True
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
        start_crawl_date: Optional[datetime] = SchemaField(
            description="Start date for crawled content", advanced=True, default=None
        )
        end_crawl_date: Optional[datetime] = SchemaField(
            description="End date for crawled content", advanced=True, default=None
        )
        start_published_date: Optional[datetime] = SchemaField(
            description="Start date for published content", advanced=True, default=None
        )
        end_published_date: Optional[datetime] = SchemaField(
            description="End date for published content", advanced=True, default=None
        )
        include_text: list[str] = SchemaField(
            description="Text patterns to include (max 1 string, up to 5 words)",
            default_factory=list,
            advanced=True,
        )
        exclude_text: list[str] = SchemaField(
            description="Text patterns to exclude (max 1 string, up to 5 words)",
            default_factory=list,
            advanced=True,
        )
        contents: ContentSettings = SchemaField(
            description="Content retrieval settings",
            default=ContentSettings(),
            advanced=True,
        )
        moderation: bool = SchemaField(
            description="Enable content moderation to filter unsafe content from search results",
            default=False,
            advanced=True,
        )

    class Output(BlockSchema):
        results: list[ExaSearchResults] = SchemaField(
            description="List of similar documents with metadata and content"
        )
        result: ExaSearchResults = SchemaField(
            description="Single similar document result",
        )
        context: str = SchemaField(
            description="A formatted string of the results ready for LLMs.",
        )
        request_id: str = SchemaField(description="Unique identifier for the request")
        cost_dollars: Optional[CostDollars] = SchemaField(
            description="Cost breakdown for the request", default=None
        )
        error: str = SchemaField(
            description="Error message if the request failed",
        )

    def __init__(self):
        super().__init__(
            id="5e7315d1-af61-4a0c-9350-7c868fa7438a",
            description="Finds similar links using Exa's findSimilar API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaFindSimilarBlock.Input,
            output_schema=ExaFindSimilarBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        url = "https://api.exa.ai/findSimilar"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        payload = {
            "url": input_data.url,
            "numResults": input_data.number_of_results,
        }

        # Handle contents field with helper function
        content_settings = process_contents_settings(input_data.contents)
        if content_settings:
            payload["contents"] = content_settings

        # Handle date fields with helper function
        date_field_mapping = {
            "start_crawl_date": "startCrawlDate",
            "end_crawl_date": "endCrawlDate",
            "start_published_date": "startPublishedDate",
            "end_published_date": "endPublishedDate",
        }
        payload.update(format_date_fields(input_data, date_field_mapping))

        # Handle other optional fields
        optional_field_mapping = {
            "include_domains": "includeDomains",
            "exclude_domains": "excludeDomains",
            "include_text": "includeText",
            "exclude_text": "excludeText",
        }
        add_optional_fields(input_data, optional_field_mapping, payload)

        # Add moderation field
        if input_data.moderation:
            payload["moderation"] = input_data.moderation

        # Always enable context for LLM-ready output
        payload["context"] = True

        try:
            response = await Requests().post(url, headers=headers, json=payload)
            data = response.json()

            # Extract all response fields
            yield "results", data.get("results", [])
            for result in data.get("results", []):
                yield "result", result

            # Yield context if present
            if "context" in data:
                yield "context", data["context"]

            # Yield request ID if present
            if "requestId" in data:
                yield "request_id", data["requestId"]

            # Yield cost information if present
            if "costDollars" in data:
                yield "cost_dollars", data["costDollars"]

        except Exception as e:
            yield "error", str(e)
