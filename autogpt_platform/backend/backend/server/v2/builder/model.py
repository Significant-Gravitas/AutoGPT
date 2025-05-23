from typing import Any, Literal

from pydantic import BaseModel

import backend.server.model as server_model
import backend.server.v2.library.model as library_model
import backend.server.v2.store.model as store_model
from backend.integrations.providers import ProviderName

FilterType = (
    Literal["all_blocks"]
    | Literal["input_blocks"]
    | Literal["action_blocks"]
    | Literal["output_blocks"]
    | Literal["integrations"]
    | Literal["providers"]
    | Literal["marketplace_agents"]
    | Literal["my_agents"]
)


SearchResultType = (
    Literal["blocks"]
    | Literal["integrations"]
    | Literal["providers"]
    | Literal["marketplace_agents"]
    | Literal["my_agents"]
)

BlockData = dict[str, Any]


class SearchOptions(BaseModel):
    search_query: str | None = None
    filter: list[FilterType] | None = None
    providers: list[str] | None = None
    by_creator: list[str] | None = None
    search_id: str | None = None
    page: int | None = None
    page_size: int | None = None


class Provider(BaseModel):
    name: ProviderName
    description: str
    integration_count: int


class BlockResponse(BaseModel):
    blocks: list[BlockData]
    total_block_count: int
    total_integration_count: int
    pagination: server_model.Pagination


class ProviderResponse(BaseModel):
    providers: list[Provider]
    pagination: server_model.Pagination


class BlockSearchResponse(BaseModel):
    items: list[
        BlockData | Provider | library_model.LibraryAgent | store_model.StoreAgent
    ]
    total_items: dict[SearchResultType, int]
    page: int
    more_pages: bool
