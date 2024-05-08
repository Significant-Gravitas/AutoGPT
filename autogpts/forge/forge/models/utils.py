from abc import ABC, abstractmethod

from pydantic import BaseModel


class ModelWithSummary(BaseModel, ABC):
    @abstractmethod
    def summary(self) -> str:
        """Should produce a human readable summary of the model content."""
        pass


class Pagination(BaseModel):
    total_items: int = Field(..., description="Total number of items.", example=42)
    total_pages: int = Field(..., description="Total number of pages.", example=97)
    current_page: int = Field(..., description="Current_page page number.", example=1)
    page_size: int = Field(..., description="Number of items per page.", example=25)
