from datetime import datetime
from typing import Optional

from exa_py import AsyncExa

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
    ContentSettings,
    CostDollars,
    ExaSearchResults,
    process_contents_settings,
)


class ExaFindSimilarBlock(Block):
    class Input(BlockSchemaInput):
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

    class Output(BlockSchemaOutput):
        results: list[ExaSearchResults] = SchemaField(
            description="List of similar documents with metadata and content"
        )
        result: ExaSearchResults = SchemaField(
            description="Single similar document result"
        )
        context: str = SchemaField(
            description="A formatted string of the results ready for LLMs."
        )
        request_id: str = SchemaField(description="Unique identifier for the request")
        cost_dollars: Optional[CostDollars] = SchemaField(
            description="Cost breakdown for the request"
        )
        error: str = SchemaField(description="Error message if the request failed")

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
        sdk_kwargs = {
            "url": input_data.url,
            "num_results": input_data.number_of_results,
        }

        # Handle domains
        if input_data.include_domains:
            sdk_kwargs["include_domains"] = input_data.include_domains
        if input_data.exclude_domains:
            sdk_kwargs["exclude_domains"] = input_data.exclude_domains

        # Handle dates
        if input_data.start_crawl_date:
            sdk_kwargs["start_crawl_date"] = input_data.start_crawl_date.isoformat()
        if input_data.end_crawl_date:
            sdk_kwargs["end_crawl_date"] = input_data.end_crawl_date.isoformat()
        if input_data.start_published_date:
            sdk_kwargs["start_published_date"] = (
                input_data.start_published_date.isoformat()
            )
        if input_data.end_published_date:
            sdk_kwargs["end_published_date"] = input_data.end_published_date.isoformat()

        # Handle text filters
        if input_data.include_text:
            sdk_kwargs["include_text"] = input_data.include_text
        if input_data.exclude_text:
            sdk_kwargs["exclude_text"] = input_data.exclude_text

        if input_data.moderation:
            sdk_kwargs["moderation"] = input_data.moderation

        # check if we need to use find_similar_and_contents
        content_settings = process_contents_settings(input_data.contents)

        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        if content_settings:
            # Use find_similar_and_contents when contents are requested
            sdk_kwargs["text"] = content_settings.get("text", False)
            if "highlights" in content_settings:
                sdk_kwargs["highlights"] = content_settings["highlights"]
            if "summary" in content_settings:
                sdk_kwargs["summary"] = content_settings["summary"]
            response = await aexa.find_similar_and_contents(**sdk_kwargs)
        else:
            response = await aexa.find_similar(**sdk_kwargs)

        converted_results = [
            ExaSearchResults.from_sdk(sdk_result)
            for sdk_result in response.results or []
        ]

        yield "results", converted_results
        for result in converted_results:
            yield "result", result

        if response.context:
            yield "context", response.context

        if response.cost_dollars:
            yield "cost_dollars", response.cost_dollars
