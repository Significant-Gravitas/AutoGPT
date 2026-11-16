"""
Shared models and types used across the backend to avoid circular imports.
"""

import pydantic

# Keeps `(page - 1) * page_size` far inside the 64-bit offset a query can carry.
MAX_PAGE = 1_000_000
# Above the largest page our frontend asks for: 1,000 store agents, on the marketplace home.
MAX_PAGE_SIZE = 1_000


class Pagination(pydantic.BaseModel):
    total_items: int = pydantic.Field(
        description="Total number of items.", examples=[42]
    )
    total_pages: int = pydantic.Field(
        description="Total number of pages.", examples=[2]
    )
    current_page: int = pydantic.Field(
        description="Current_page page number.", examples=[1]
    )
    page_size: int = pydantic.Field(
        description="Number of items per page.", examples=[25]
    )

    @staticmethod
    def empty() -> "Pagination":
        return Pagination(
            total_items=0,
            total_pages=0,
            current_page=0,
            page_size=0,
        )
