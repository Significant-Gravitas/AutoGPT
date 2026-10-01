from datetime import date
from typing import Literal, TypedDict

from backend.sdk import BaseModel, SchemaField

LinkupSearchDepth = Literal["fast", "standard", "deep"]
LinkupOutputType = Literal["searchResults", "sourcedAnswer"]


class LinkupSearchParams(TypedDict):
    """Keyword arguments shared by every ``LinkupClient.async_search`` call."""

    query: str
    depth: LinkupSearchDepth
    max_results: int
    include_domains: list[str] | None
    exclude_domains: list[str] | None
    from_date: date | None
    to_date: date | None


class LinkupSearchResult(BaseModel):
    """Schema for a single Linkup search result."""

    name: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    content: str = SchemaField(description="Content extracted from the page")


class LinkupAnswerSource(BaseModel):
    """Schema for a source supporting a Linkup answer."""

    name: str = SchemaField(description="Title of the source page")
    url: str = SchemaField(description="URL of the source page")
    snippet: str = SchemaField(
        description="Excerpt from the source that supports the answer"
    )


def search_cost_usd(depth: LinkupSearchDepth, output_type: LinkupOutputType) -> float:
    """USD price of one /search call per Linkup's published pricing."""
    deep = depth == "deep"
    if output_type == "sourcedAnswer":
        return 0.055 if deep else 0.006
    return 0.05 if deep else 0.005


def fetch_cost_usd(render_js: bool) -> float:
    """USD price of one /fetch call (standard mode) per Linkup's published pricing."""
    return 0.005 if render_js else 0.001
