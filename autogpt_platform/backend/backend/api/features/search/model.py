"""Pydantic response models for the unified ``/api/search`` endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import pydantic

SearchItemType = Literal[
    "library_agent",
    "store_agent",
    "workspace_file",
    "chat_session",
]


class SearchResultItem(pydantic.BaseModel):
    """One row in a /search/global bucket.

    Kept intentionally generic so the frontend can render all three
    buckets through a single component. Type-specific details live in
    ``metadata`` (e.g. mime_type for files, creator for store agents).
    """

    id: str
    type: SearchItemType
    title: str
    subtitle: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)
    # ``None`` for empty-query (recent) responses — no relevance ranking,
    # just newest-first. Populated for hybrid-search responses.
    score: float | None = None
    updated_at: datetime | None = None


class GlobalSearchResponse(pydantic.BaseModel):
    """Top-N hits bucketed by item type.

    For a non-empty query this contains hybrid-search results.
    For an empty query this contains the most-recently-updated items
    per bucket (cached per-user with a short TTL).
    """

    agents: list[SearchResultItem] = pydantic.Field(default_factory=list)
    files: list[SearchResultItem] = pydantic.Field(default_factory=list)
    chats: list[SearchResultItem] = pydantic.Field(default_factory=list)
