from typing import Literal

from pydantic import BaseModel

import backend.api.features.library.model as library_model
import backend.api.features.store.model as store_model
from backend.blocks._base import BlockInfo
from backend.integrations.providers import ProviderName
from backend.util.models import Pagination

FilterType = Literal[
    "blocks",
    "integrations",
    "marketplace_agents",
    "my_agents",
]

BlockTypeFilter = Literal["all", "input", "action", "output"]


class SearchEntry(BaseModel):
    search_query: str | None = None
    filter: list[FilterType] | None = None
    by_creator: list[str] | None = None
    search_id: str | None = None


# Suggestions
class SuggestionsResponse(BaseModel):
    recent_searches: list[SearchEntry]
    providers: list[ProviderName]
    top_blocks: list[BlockInfo]


# All blocks
class BlockCategoryResponse(BaseModel):
    name: str
    total_blocks: int
    blocks: list[BlockInfo]

    model_config = {"use_enum_values": False}  # Use enum names like "AI"


# Input/Action/Output and see all for block categories
class BlockResponse(BaseModel):
    blocks: list[BlockInfo]
    pagination: Pagination


# Providers
class Provider(BaseModel):
    name: ProviderName
    description: str
    integration_count: int


class ProviderResponse(BaseModel):
    providers: list[Provider]
    pagination: Pagination


class SearchResponse(BaseModel):
    items: list[BlockInfo | library_model.LibraryAgent | store_model.StoreAgent]
    search_id: str
    total_items: dict[FilterType, int]
    pagination: Pagination


class CountResponse(BaseModel):
    all_blocks: int
    input_blocks: int
    action_blocks: int
    output_blocks: int
    integrations: int
    marketplace_agents: int
    my_agents: int
