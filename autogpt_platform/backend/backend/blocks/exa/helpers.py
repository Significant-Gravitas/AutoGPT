from typing import Any, Optional

from pydantic import BaseModel

from backend.data.model import SchemaField


def _to_camel_case(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def to_camel_case_dict(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, val in data.items():
        camel_key = _to_camel_case(key)
        if isinstance(val, dict):
            result[camel_key] = to_camel_case_dict(val)
        elif isinstance(val, list):
            result[camel_key] = [
                to_camel_case_dict(v) if isinstance(v, dict) else v for v in val
            ]
        else:
            result[camel_key] = val
    return result


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
