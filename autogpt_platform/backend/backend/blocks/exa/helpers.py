from typing import Optional

from backend.sdk import BaseModel, SchemaField


class TextSettings(BaseModel):
    max_characters: int = SchemaField(
        default=1000,
        description="Maximum number of characters to return",
        placeholder="1000",
    )
    include_html_tags: bool = SchemaField(
        default=False,
        description="Whether to include HTML tags in the text",
        placeholder="False",
    )


class HighlightSettings(BaseModel):
    num_sentences: int = SchemaField(
        default=3,
        description="Number of sentences per highlight",
        placeholder="3",
    )
    highlights_per_url: int = SchemaField(
        default=3,
        description="Number of highlights per URL",
        placeholder="3",
    )


class SummarySettings(BaseModel):
    query: Optional[str] = SchemaField(
        default="",
        description="Query string for summarization",
        placeholder="Enter query",
    )


class ContentSettings(BaseModel):
    text: TextSettings = SchemaField(
        default=TextSettings(),
    )
    highlights: HighlightSettings = SchemaField(
        default=HighlightSettings(),
    )
    summary: SummarySettings = SchemaField(
        default=SummarySettings(),
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
