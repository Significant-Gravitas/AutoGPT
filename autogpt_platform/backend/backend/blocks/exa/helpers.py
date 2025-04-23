from typing import Optional

from pydantic import BaseModel

from backend.data.model import SchemaField


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
        description="Text content settings",
    )
    highlights: HighlightSettings = SchemaField(
        default=HighlightSettings(),
        description="Highlight settings",
    )
    summary: SummarySettings = SchemaField(
        default=SummarySettings(),
        description="Summary settings",
    )
