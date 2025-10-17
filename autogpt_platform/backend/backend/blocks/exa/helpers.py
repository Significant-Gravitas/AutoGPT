from enum import Enum
from typing import Optional, Union

from backend.sdk import BaseModel, SchemaField


class LivecrawlTypes(str, Enum):
    NEVER = "never"
    FALLBACK = "fallback"
    ALWAYS = "always"
    PREFERRED = "preferred"


class TextSettings(BaseModel):
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


class ContextSettings(BaseModel):
    max_characters: Optional[int] = SchemaField(
        default=None,
        description="Maximum character limit for context string",
        placeholder="10000",
    )


class ContentSettings(BaseModel):
    text: Optional[Union[bool, TextSettings]] = SchemaField(
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
    context: Optional[Union[bool, ContextSettings]] = SchemaField(
        default=None,
        description="Format search results into a context string for LLMs",
        advanced=True,
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
