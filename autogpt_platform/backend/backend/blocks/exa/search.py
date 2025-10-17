from datetime import datetime
from enum import Enum

from pydantic import BaseModel

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    MediaFileType,
    Requests,
    SchemaField,
)

from ._config import exa
from .helpers import ContentSettings


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


class ExaSearchExtras(BaseModel):
    links: list[str] = SchemaField(
        default_factory=list, description="Array of links from the search result"
    )
    imageLinks: list[str] = SchemaField(
        default_factory=list, description="Array of image links from the search result"
    )


class CostBreakdown(BaseModel):
    keywordSearch: float = SchemaField(default=0.0)
    neuralSearch: float = SchemaField(default=0.0)
    contentText: float = SchemaField(default=0.0)
    contentHighlight: float = SchemaField(default=0.0)
    contentSummary: float = SchemaField(default=0.0)


class CostBreakdownItem(BaseModel):
    search: float = SchemaField(default=0.0)
    contents: float = SchemaField(default=0.0)
    breakdown: CostBreakdown = SchemaField(default_factory=CostBreakdown)


class PerRequestPrices(BaseModel):
    neuralSearch_1_25_results: float = SchemaField(default=0.005)
    neuralSearch_26_100_results: float = SchemaField(default=0.025)
    neuralSearch_100_plus_results: float = SchemaField(default=1.0)
    keywordSearch_1_100_results: float = SchemaField(default=0.0025)
    keywordSearch_100_plus_results: float = SchemaField(default=3.0)


class PerPagePrices(BaseModel):
    contentText: float = SchemaField(default=0.001)
    contentHighlight: float = SchemaField(default=0.001)
    contentSummary: float = SchemaField(default=0.001)


class CostDollars(BaseModel):
    total: float = SchemaField(description="Total dollar cost for your request")
    breakDown: list[CostBreakdownItem] = SchemaField(
        default_factory=list, description="Breakdown of costs by operation type"
    )
    perRequestPrices: PerRequestPrices = SchemaField(
        default_factory=PerRequestPrices,
        description="Standard price per request for different operations",
    )
    perPagePrices: PerPagePrices = SchemaField(
        default_factory=PerPagePrices,
        description="Standard price per page for different content operations",
    )


class ExaSearchResults(BaseModel):
    title: str | None = None
    url: str | None = None
    publishedDate: str | None = None
    author: str | None = None
    id: str
    image: MediaFileType | None = None
    favicon: MediaFileType | None = None
    text: str | None = None
    highlights: list[str] = SchemaField(default_factory=list)
    highlightScores: list[float] = SchemaField(default_factory=list)
    summary: str | None = None
    subpages: list[dict] = SchemaField(default_factory=list)
    extras: ExaSearchExtras | None = None


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
        context: bool | dict = SchemaField(
            description="Formats the search results into a context string ready for LLMs. Can be boolean or object with maxCharacters",
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
        cost_dollars: CostDollars | None = SchemaField(
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

        # Handle contents field with new structure
        if input_data.contents:
            content_settings = {}

            # Handle text field (can be boolean or object)
            if input_data.contents.text is not None:
                if isinstance(input_data.contents.text, bool):
                    content_settings["text"] = input_data.contents.text
                else:
                    text_dict = {}
                    if input_data.contents.text.max_characters:
                        text_dict["maxCharacters"] = (
                            input_data.contents.text.max_characters
                        )
                    if input_data.contents.text.include_html_tags:
                        text_dict["includeHtmlTags"] = (
                            input_data.contents.text.include_html_tags
                        )
                    content_settings["text"] = text_dict

            # Handle highlights
            if input_data.contents.highlights:
                highlights_dict = {}
                highlights_dict["numSentences"] = (
                    input_data.contents.highlights.num_sentences
                )
                highlights_dict["highlightsPerUrl"] = (
                    input_data.contents.highlights.highlights_per_url
                )
                if input_data.contents.highlights.query:
                    highlights_dict["query"] = input_data.contents.highlights.query
                content_settings["highlights"] = highlights_dict

            # Handle summary
            if input_data.contents.summary:
                summary_dict = {}
                if input_data.contents.summary.query:
                    summary_dict["query"] = input_data.contents.summary.query
                if input_data.contents.summary.schema:
                    summary_dict["schema"] = input_data.contents.summary.schema
                content_settings["summary"] = summary_dict

            # Handle livecrawl
            if input_data.contents.livecrawl:
                content_settings["livecrawl"] = input_data.contents.livecrawl.value

            # Handle livecrawl_timeout
            if input_data.contents.livecrawl_timeout is not None:
                content_settings["livecrawlTimeout"] = (
                    input_data.contents.livecrawl_timeout
                )

            # Handle subpages
            if input_data.contents.subpages is not None:
                content_settings["subpages"] = input_data.contents.subpages

            # Handle subpage_target
            if input_data.contents.subpage_target:
                content_settings["subpageTarget"] = input_data.contents.subpage_target

            # Handle extras
            if input_data.contents.extras:
                extras_dict = {}
                if input_data.contents.extras.links:
                    extras_dict["links"] = input_data.contents.extras.links
                if input_data.contents.extras.image_links:
                    extras_dict["imageLinks"] = input_data.contents.extras.image_links
                content_settings["extras"] = extras_dict

            # Handle context within contents
            if input_data.contents.context is not None:
                if isinstance(input_data.contents.context, bool):
                    content_settings["context"] = input_data.contents.context
                else:
                    context_dict = {}
                    if input_data.contents.context.max_characters:
                        context_dict["maxCharacters"] = (
                            input_data.contents.context.max_characters
                        )
                    content_settings["context"] = context_dict

            if content_settings:  # Only add if there are actual settings
                payload["contents"] = content_settings

        date_field_mapping = {
            "start_crawl_date": "startCrawlDate",
            "end_crawl_date": "endCrawlDate",
            "start_published_date": "startPublishedDate",
            "end_published_date": "endPublishedDate",
        }

        # Add dates if they exist
        for input_field, api_field in date_field_mapping.items():
            value = getattr(input_data, input_field, None)
            if value:
                payload[api_field] = value.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        optional_field_mapping = {
            "type": "type",
            "category": "category",
            "user_location": "userLocation",
            "include_domains": "includeDomains",
            "exclude_domains": "excludeDomains",
            "include_text": "includeText",
            "exclude_text": "excludeText",
        }

        # Add other fields
        for input_field, api_field in optional_field_mapping.items():
            value = getattr(input_data, input_field)
            if value:  # Only add non-empty values
                if input_field == "type" or input_field == "category":
                    payload[api_field] = (
                        value.value if hasattr(value, "value") else value
                    )
                else:
                    payload[api_field] = value

        # Add moderation field
        if input_data.moderation:
            payload["moderation"] = input_data.moderation

        # Add context field (from Input, not from contents)
        if input_data.context:
            if isinstance(input_data.context, bool):
                payload["context"] = input_data.context
            elif (
                isinstance(input_data.context, dict)
                and "maxCharacters" in input_data.context
            ):
                payload["context"] = {
                    "maxCharacters": input_data.context["maxCharacters"]
                }

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
