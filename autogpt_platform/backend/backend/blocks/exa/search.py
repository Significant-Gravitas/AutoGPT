from datetime import datetime
from enum import Enum
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
    class Input(BlockSchema):
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

    class Output(BlockSchema):
        results: list[ExaSearchResults] = SchemaField(
            description="List of search results"
        )
        result: ExaSearchResults = SchemaField(
            description="Single search result",
        )
        context: str = SchemaField(
            description="A formatted string of the search results ready for LLMs.",
        )
        search_type: str = SchemaField(
            description="For auto searches, indicates which search type was selected."
        )
        request_id: str = SchemaField(description="Unique identifier for the request")
        resolved_search_type: str = SchemaField(
            description="The search type that was actually used for this request (neural or keyword)"
        )
        cost_dollars: Optional[CostDollars] = SchemaField(
            description="Cost breakdown for the request", default=None
        )
        error: str = SchemaField(
            description="Error message if the request failed",
        )

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
        url = "https://api.exa.ai/search"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": credentials.api_key.get_secret_value(),
        }

        payload = {
            "query": input_data.query,
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

        # Handle enum fields separately since they need special processing
        for field_name, api_field in [("type", "type"), ("category", "category")]:
            value = getattr(input_data, field_name, None)
            if value:
                payload[api_field] = value.value if hasattr(value, "value") else value

        # Handle other optional fields
        optional_field_mapping = {
            "user_location": "userLocation",
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

            # Yield search type if present
            if "searchType" in data:
                yield "search_type", data["searchType"]

            # Yield request ID if present
            if "requestId" in data:
                yield "request_id", data["requestId"]

            # Yield resolved search type if present
            if "resolvedSearchType" in data:
                yield "resolved_search_type", data["resolvedSearchType"]

            # Yield cost information if present
            if "costDollars" in data:
                yield "cost_dollars", data["costDollars"]

        except Exception as e:
            yield "error", str(e)
