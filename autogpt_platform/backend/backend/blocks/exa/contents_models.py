"""Response models for the Exa Contents block (``/contents``)."""

from enum import Enum

from pydantic import BaseModel

from backend.sdk import SchemaField

from .helpers import ExaSearchResults


class ContentStatusTag(str, Enum):
    CRAWL_NOT_FOUND = "CRAWL_NOT_FOUND"
    CRAWL_TIMEOUT = "CRAWL_TIMEOUT"
    CRAWL_LIVECRAWL_TIMEOUT = "CRAWL_LIVECRAWL_TIMEOUT"
    SOURCE_NOT_AVAILABLE = "SOURCE_NOT_AVAILABLE"
    CRAWL_UNKNOWN_ERROR = "CRAWL_UNKNOWN_ERROR"


class ContentError(BaseModel):
    tag: ContentStatusTag | str | None = SchemaField(
        default=None,
        description="Specific error type, e.g. CRAWL_NOT_FOUND (Exa may add new tags)",
    )
    httpStatusCode: int | None = SchemaField(
        default=None, description="The corresponding HTTP status code"
    )


class ContentStatus(BaseModel):
    id: str = SchemaField(description="The URL that was requested")
    status: str = SchemaField(
        description="Status of the content fetch operation (success or error)"
    )
    source: str | None = SchemaField(
        default=None,
        description="Where the content came from: 'cached' or 'crawled'",
    )
    error: ContentError | None = SchemaField(
        default=None, description="Error details, only present when status is 'error'"
    )


class ExaContentsResponse(BaseModel):
    requestId: str = ""
    results: list[ExaSearchResults] = []
    statuses: list[ContentStatus] = []
    context: str | None = None
