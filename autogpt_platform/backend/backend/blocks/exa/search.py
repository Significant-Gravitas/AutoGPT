from datetime import datetime
from enum import Enum
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


class ExaSearchTypes(Enum):
    KEYWORD = "keyword"
    NEURAL = "neural"
    FAST = "fast"
    AUTO = "auto"


class ExaSearchCategories(Enum):
    COMPANY = "company"
    RESEARCH_PAPER = "research paper"
    NEWS = "news"
    PDF = "pdf"
    GITHUB = "github"
    TWEET = "tweet"
    PERSONAL_SITE = "personal site"
    LINKEDIN_PROFILE = "linkedin profile"
    FINANCIAL_REPORT = "financial report"


class ExaSearchBlock(Block):
    class Input(BlockSchemaInput):
        credentials: CredentialsMetaInput = exa.credentials_field(
            description="The Exa integration requires an API Key."
        )
        query: str = SchemaField(description="The search query")
        type: ExaSearchTypes = SchemaField(
            description="Type of search", default=ExaSearchTypes.AUTO, advanced=True
        )
        category: ExaSearchCategories | None = SchemaField(
            description="Category to search within: company, research paper, news, pdf, github, tweet, personal site, linkedin profile, financial report",
            default=None,
            advanced=True,
        )
        user_location: str | None = SchemaField(
            description="The two-letter ISO country code of the user (e.g., 'US')",
            default=None,
            advanced=True,
        )
        number_of_results: int = SchemaField(
            description="Number of results to return", default=10, advanced=True
        )
        include_domains: list[str] = SchemaField(
            description="Domains to include in search", default_factory=list
        )
        exclude_domains: list[str] = SchemaField(
            description="Domains to exclude from search",
            default_factory=list,
            advanced=True,
        )
        start_crawl_date: datetime | None = SchemaField(
            description="Start date for crawled content", advanced=True, default=None
        )
        end_crawl_date: datetime | None = SchemaField(
            description="End date for crawled content", advanced=True, default=None
        )
        start_published_date: datetime | None = SchemaField(
            description="Start date for published content", advanced=True, default=None
        )
        end_published_date: datetime | None = SchemaField(
            description="End date for published content", advanced=True, default=None
        )
        include_text: list[str] = SchemaField(
            description="Text patterns to include", default_factory=list, advanced=True
        )
        exclude_text: list[str] = SchemaField(
            description="Text patterns to exclude", default_factory=list, advanced=True
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
            description="List of search results"
        )
        result: ExaSearchResults = SchemaField(description="Single search result")
        context: str = SchemaField(
            description="A formatted string of the search results ready for LLMs."
        )
        search_type: str = SchemaField(
            description="For auto searches, indicates which search type was selected."
        )
        resolved_search_type: str = SchemaField(
            description="The search type that was actually used for this request (neural or keyword)"
        )
        cost_dollars: Optional[CostDollars] = SchemaField(
            description="Cost breakdown for the request"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="996cec64-ac40-4dde-982f-b0dc60a5824d",
            description="Searches the web using Exa's advanced search API",
            categories={BlockCategory.SEARCH},
            input_schema=ExaSearchBlock.Input,
            output_schema=ExaSearchBlock.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        sdk_kwargs = {
            "query": input_data.query,
            "num_results": input_data.number_of_results,
        }

        if input_data.type:
            sdk_kwargs["type"] = input_data.type.value

        if input_data.category:
            sdk_kwargs["category"] = input_data.category.value

        if input_data.user_location:
            sdk_kwargs["user_location"] = input_data.user_location

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

        # heck if we need to use search_and_contents
        content_settings = process_contents_settings(input_data.contents)

        aexa = AsyncExa(api_key=credentials.api_key.get_secret_value())

        if content_settings:
            sdk_kwargs["text"] = content_settings.get("text", False)
            if "highlights" in content_settings:
                sdk_kwargs["highlights"] = content_settings["highlights"]
            if "summary" in content_settings:
                sdk_kwargs["summary"] = content_settings["summary"]
            response = await aexa.search_and_contents(**sdk_kwargs)
        else:
            response = await aexa.search(**sdk_kwargs)

        converted_results = [
            ExaSearchResults.from_sdk(sdk_result)
            for sdk_result in response.results or []
        ]

        yield "results", converted_results
        for result in converted_results:
            yield "result", result

        if response.context:
            yield "context", response.context

        if response.resolved_search_type:
            yield "resolved_search_type", response.resolved_search_type

        if response.cost_dollars:
            yield "cost_dollars", response.cost_dollars
