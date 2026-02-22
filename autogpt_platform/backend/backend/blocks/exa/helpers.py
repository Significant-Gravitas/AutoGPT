from enum import Enum
from typing import Any, Dict, Literal, Optional, Union

from backend.sdk import BaseModel, MediaFileType, SchemaField


class LivecrawlTypes(str, Enum):
    NEVER = "never"
    FALLBACK = "fallback"
    ALWAYS = "always"
    PREFERRED = "preferred"


class TextEnabled(BaseModel):
    discriminator: Literal["enabled"] = "enabled"


class TextDisabled(BaseModel):
    discriminator: Literal["disabled"] = "disabled"


class TextAdvanced(BaseModel):
    discriminator: Literal["advanced"] = "advanced"
    max_characters: Optional[int] = SchemaField(
        default=None,
        description="Maximum number of characters to return",
        placeholder="1000",
    )
    include_html_tags: bool = SchemaField(
        default=False,
        description="Include HTML tags in the response, helps LLMs understand text structure",
        placeholder="False",
    )


class HighlightSettings(BaseModel):
    num_sentences: int = SchemaField(
        default=1,
        description="Number of sentences per highlight",
        placeholder="1",
        ge=1,
    )
    highlights_per_url: int = SchemaField(
        default=1,
        description="Number of highlights per URL",
        placeholder="1",
        ge=1,
    )
    query: Optional[str] = SchemaField(
        default=None,
        description="Custom query to direct the LLM's selection of highlights",
        placeholder="Key advancements",
    )


class SummarySettings(BaseModel):
    query: Optional[str] = SchemaField(
        default=None,
        description="Custom query for the LLM-generated summary",
        placeholder="Main developments",
    )
    schema: Optional[dict] = SchemaField(  # type: ignore
        default=None,
        description="JSON schema for structured output from summary",
        advanced=True,
    )


class ExtrasSettings(BaseModel):
    links: int = SchemaField(
        default=0,
        description="Number of URLs to return from each webpage",
        placeholder="1",
        ge=0,
    )
    image_links: int = SchemaField(
        default=0,
        description="Number of images to return for each result",
        placeholder="1",
        ge=0,
    )


class ContextEnabled(BaseModel):
    discriminator: Literal["enabled"] = "enabled"


class ContextDisabled(BaseModel):
    discriminator: Literal["disabled"] = "disabled"


class ContextAdvanced(BaseModel):
    discriminator: Literal["advanced"] = "advanced"
    max_characters: Optional[int] = SchemaField(
        default=None,
        description="Maximum character limit for context string",
        placeholder="10000",
    )


class ContentSettings(BaseModel):
    text: Optional[Union[bool, TextEnabled, TextDisabled, TextAdvanced]] = SchemaField(
        default=None,
        description="Text content retrieval. Boolean for simple enable/disable or object for advanced settings",
    )
    highlights: Optional[HighlightSettings] = SchemaField(
        default=None,
        description="Text snippets most relevant from each page",
    )
    summary: Optional[SummarySettings] = SchemaField(
        default=None,
        description="LLM-generated summary of the webpage",
    )
    livecrawl: Optional[LivecrawlTypes] = SchemaField(
        default=None,
        description="Livecrawling options: never, fallback, always, preferred",
        advanced=True,
    )
    livecrawl_timeout: Optional[int] = SchemaField(
        default=None,
        description="Timeout for livecrawling in milliseconds",
        placeholder="10000",
        advanced=True,
    )
    subpages: Optional[int] = SchemaField(
        default=None,
        description="Number of subpages to crawl",
        placeholder="0",
        ge=0,
        advanced=True,
    )
    subpage_target: Optional[Union[str, list[str]]] = SchemaField(
        default=None,
        description="Keyword(s) to find specific subpages of search results",
        advanced=True,
    )
    extras: Optional[ExtrasSettings] = SchemaField(
        default=None,
        description="Extra parameters for additional content",
        advanced=True,
    )
    context: Optional[Union[bool, ContextEnabled, ContextDisabled, ContextAdvanced]] = (
        SchemaField(
            default=None,
            description="Format search results into a context string for LLMs",
            advanced=True,
        )
    )


# Websets Models
class WebsetEntitySettings(BaseModel):
    type: Optional[str] = SchemaField(
        default=None,
        description="Entity type (e.g., 'company', 'person')",
        placeholder="company",
    )


class WebsetCriterion(BaseModel):
    description: str = SchemaField(
        description="Description of the criterion",
        placeholder="Must be based in the US",
    )
    success_rate: Optional[int] = SchemaField(
        default=None,
        description="Success rate percentage",
        ge=0,
        le=100,
    )


class WebsetSearchConfig(BaseModel):
    query: str = SchemaField(
        description="Search query",
        placeholder="Marketing agencies based in the US",
    )
    count: int = SchemaField(
        default=10,
        description="Number of results to return",
        ge=1,
        le=100,
    )
    entity: Optional[WebsetEntitySettings] = SchemaField(
        default=None,
        description="Entity settings for the search",
    )
    criteria: Optional[list[WebsetCriterion]] = SchemaField(
        default=None,
        description="Search criteria",
    )
    behavior: Optional[str] = SchemaField(
        default="override",
        description="Behavior when updating results ('override' or 'append')",
        placeholder="override",
    )


class EnrichmentOption(BaseModel):
    label: str = SchemaField(
        description="Label for the enrichment option",
        placeholder="Option 1",
    )


class WebsetEnrichmentConfig(BaseModel):
    title: str = SchemaField(
        description="Title of the enrichment",
        placeholder="Company Details",
    )
    description: str = SchemaField(
        description="Description of what this enrichment does",
        placeholder="Extract company information",
    )
    format: str = SchemaField(
        default="text",
        description="Format of the enrichment result",
        placeholder="text",
    )
    instructions: Optional[str] = SchemaField(
        default=None,
        description="Instructions for the enrichment",
        placeholder="Extract key company metrics",
    )
    options: Optional[list[EnrichmentOption]] = SchemaField(
        default=None,
        description="Options for the enrichment",
    )


# Shared result models
class ExaSearchExtras(BaseModel):
    links: list[str] = SchemaField(
        default_factory=list, description="Array of links from the search result"
    )
    imageLinks: list[str] = SchemaField(
        default_factory=list, description="Array of image links from the search result"
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

    @classmethod
    def from_sdk(cls, sdk_result) -> "ExaSearchResults":
        """Convert SDK Result (dataclass) to our Pydantic model."""
        return cls(
            id=getattr(sdk_result, "id", ""),
            url=getattr(sdk_result, "url", None),
            title=getattr(sdk_result, "title", None),
            author=getattr(sdk_result, "author", None),
            publishedDate=getattr(sdk_result, "published_date", None),
            text=getattr(sdk_result, "text", None),
            highlights=getattr(sdk_result, "highlights", None) or [],
            highlightScores=getattr(sdk_result, "highlight_scores", None) or [],
            summary=getattr(sdk_result, "summary", None),
            subpages=getattr(sdk_result, "subpages", None) or [],
            image=getattr(sdk_result, "image", None),
            favicon=getattr(sdk_result, "favicon", None),
            extras=getattr(sdk_result, "extras", None),
        )


# Cost tracking models
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


# Helper functions for payload processing
def process_text_field(
    text: Union[bool, TextEnabled, TextDisabled, TextAdvanced, None]
) -> Optional[Union[bool, Dict[str, Any]]]:
    """Process text field for API payload."""
    if text is None:
        return None

    # Handle backward compatibility with boolean
    if isinstance(text, bool):
        return text
    elif isinstance(text, TextDisabled):
        return False
    elif isinstance(text, TextEnabled):
        return True
    elif isinstance(text, TextAdvanced):
        text_dict = {}
        if text.max_characters:
            text_dict["maxCharacters"] = text.max_characters
        if text.include_html_tags:
            text_dict["includeHtmlTags"] = text.include_html_tags
        return text_dict if text_dict else True
    return None


def process_contents_settings(contents: Optional[ContentSettings]) -> Dict[str, Any]:
    """Process ContentSettings into API payload format."""
    if not contents:
        return {}

    content_settings = {}

    # Handle text field (can be boolean or object)
    text_value = process_text_field(contents.text)
    if text_value is not None:
        content_settings["text"] = text_value

    # Handle highlights
    if contents.highlights:
        highlights_dict: Dict[str, Any] = {
            "numSentences": contents.highlights.num_sentences,
            "highlightsPerUrl": contents.highlights.highlights_per_url,
        }
        if contents.highlights.query:
            highlights_dict["query"] = contents.highlights.query
        content_settings["highlights"] = highlights_dict

    if contents.summary:
        summary_dict = {}
        if contents.summary.query:
            summary_dict["query"] = contents.summary.query
        if contents.summary.schema:
            summary_dict["schema"] = contents.summary.schema
        content_settings["summary"] = summary_dict

    if contents.livecrawl:
        content_settings["livecrawl"] = contents.livecrawl.value

    if contents.livecrawl_timeout is not None:
        content_settings["livecrawlTimeout"] = contents.livecrawl_timeout

    if contents.subpages is not None:
        content_settings["subpages"] = contents.subpages

    if contents.subpage_target:
        content_settings["subpageTarget"] = contents.subpage_target

    if contents.extras:
        extras_dict = {}
        if contents.extras.links:
            extras_dict["links"] = contents.extras.links
        if contents.extras.image_links:
            extras_dict["imageLinks"] = contents.extras.image_links
        content_settings["extras"] = extras_dict

    context_value = process_context_field(contents.context)
    if context_value is not None:
        content_settings["context"] = context_value

    return content_settings


def process_context_field(
    context: Union[bool, dict, ContextEnabled, ContextDisabled, ContextAdvanced, None]
) -> Optional[Union[bool, Dict[str, int]]]:
    """Process context field for API payload."""
    if context is None:
        return None

    # Handle backward compatibility with boolean
    if isinstance(context, bool):
        return context if context else None
    elif isinstance(context, dict) and "maxCharacters" in context:
        return {"maxCharacters": context["maxCharacters"]}
    elif isinstance(context, ContextDisabled):
        return None  # Don't send context field at all when disabled
    elif isinstance(context, ContextEnabled):
        return True
    elif isinstance(context, ContextAdvanced):
        if context.max_characters:
            return {"maxCharacters": context.max_characters}
        return True
    return None


def format_date_fields(
    input_data: Any, date_field_mapping: Dict[str, str]
) -> Dict[str, str]:
    """Format datetime fields for API payload."""
    formatted_dates = {}
    for input_field, api_field in date_field_mapping.items():
        value = getattr(input_data, input_field, None)
        if value:
            formatted_dates[api_field] = value.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return formatted_dates


def add_optional_fields(
    input_data: Any,
    field_mapping: Dict[str, str],
    payload: Dict[str, Any],
    process_enums: bool = False,
) -> None:
    """Add optional fields to payload if they have values."""
    for input_field, api_field in field_mapping.items():
        value = getattr(input_data, input_field, None)
        if value:  # Only add non-empty values
            if process_enums and hasattr(value, "value"):
                payload[api_field] = value.value
            else:
                payload[api_field] = value
