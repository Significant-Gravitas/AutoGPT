from enum import Enum
from typing import Any, Optional

from backend.sdk import BaseModel, SchemaField

# USD cost of one Tavily API credit on the pay-as-you-go plan.
CREDIT_USD = 0.008


def credits_from_response(response: dict[str, Any], estimate: float) -> float:
    """Actual credits consumed per the API's usage report (include_usage=True),
    falling back to the documented credit schedule when absent."""
    usage = response.get("usage") or {}
    credits = usage.get("credits")
    return credits if isinstance(credits, (int, float)) else estimate


class TavilySearchDepth(Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    FAST = "fast"
    ULTRA_FAST = "ultra-fast"


class TavilyTopic(Enum):
    GENERAL = "general"
    NEWS = "news"
    FINANCE = "finance"


class TavilyTimeRange(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class TavilyExtractDepth(Enum):
    BASIC = "basic"
    ADVANCED = "advanced"


class TavilyFormat(Enum):
    MARKDOWN = "markdown"
    TEXT = "text"


class TavilySearchResult(BaseModel):
    """Schema for a single Tavily search result."""

    title: str = SchemaField(description="Title of the result page")
    url: str = SchemaField(description="URL of the result page")
    content: str = SchemaField(
        description="Query-relevant content snippet extracted from the page"
    )
    score: float = SchemaField(description="Relevance score for the query (0-1)")
    raw_content: Optional[str] = SchemaField(
        description="Full page content, present when include_raw_content is enabled",
        default=None,
    )


class TavilyPageContent(BaseModel):
    """Schema for a page returned by Tavily extract/crawl."""

    url: str = SchemaField(description="URL of the page")
    raw_content: str = SchemaField(description="Extracted page content")
