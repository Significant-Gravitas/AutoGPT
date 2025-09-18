from typing import Literal

from pydantic import BaseModel

import backend.server.v2.library.model as library_model
import backend.server.v2.store.model as store_model
from backend.data.block import BlockInfo
from backend.integrations.providers import ProviderName
from backend.util.models import Pagination

FilterType = Literal[
    "blocks",
    "integrations",
    "marketplace_agents",
    "my_agents",
]

BlockType = Literal["all", "input", "action", "output"]


# Suggestions
class SuggestionsResponse(BaseModel):
    otto_suggestions: list[str]
    recent_searches: list[str]
    providers: list[ProviderName]
    top_blocks: list[BlockInfo]


# All blocks
class BlockCategoryResponse(BaseModel):
    name: str
    total_blocks: int
    blocks: list[BlockInfo]

    model_config = {"use_enum_values": False}  # <== use enum names like "AI"


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


# Search
class SearchRequest(BaseModel):
    search_query: str | None = None
    filter: list[FilterType] | None = None
    by_creator: list[str] | None = None
    search_id: str | None = None
    page: int | None = None
    page_size: int | None = None


class SearchBlocksResponse(BaseModel):
    blocks: BlockResponse
    total_block_count: int
    total_integration_count: int


class SearchResponse(BaseModel):
    items: list[BlockInfo | library_model.LibraryAgent | store_model.StoreAgent]
    total_items: dict[FilterType, int]
    page: int
    more_pages: bool


class CountResponse(BaseModel):
    all_blocks: int
    input_blocks: int
    action_blocks: int
    output_blocks: int
    integrations: int
    marketplace_agents: int
    my_agents: int
