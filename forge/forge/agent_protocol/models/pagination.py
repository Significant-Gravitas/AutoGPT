from pydantic import BaseModel, Field


class Pagination(BaseModel):
    total_items: int = Field(description="Total number of items.", examples=[42])
    total_pages: int = Field(description="Total number of pages.", examples=[97])
    current_page: int = Field(description="Current_page page number.", examples=[1])
    page_size: int = Field(description="Number of items per page.", examples=[25])
