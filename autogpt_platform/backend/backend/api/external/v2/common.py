"""
Common utilities for V2 External API
"""

from typing import TypeVar

from pydantic import BaseModel, Field

# Constants for pagination
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20


class PaginationParams(BaseModel):
    """Common pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Number of items per page (max {MAX_PAGE_SIZE})",
    )


T = TypeVar("T")


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""

    items: list
    total_count: int = Field(description="Total number of items across all pages")
    page: int = Field(description="Current page number (1-indexed)")
    page_size: int = Field(description="Number of items per page")
    total_pages: int = Field(description="Total number of pages")
