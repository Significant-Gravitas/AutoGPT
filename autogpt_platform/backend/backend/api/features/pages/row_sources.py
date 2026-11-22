"""Where a section's rows come from. Sections read through ``RowSource`` and
never the collection tables, so a connected source is one more implementation."""

from typing import Protocol

from backend.data.collections.model import (
    Aggregate,
    CollectionField,
    CollectionInfo,
    Condition,
    RowPage,
    RowQuery,
)
from backend.data.db_accessors import collections_db


class RowSource(Protocol):
    id: str
    name: str
    fields: list[CollectionField]
    key_field: str | None
    # Whether the owner may edit, add and delete rows through a page.
    writable: bool

    async def query(self, user_id: str, query: RowQuery) -> RowPage: ...

    async def aggregate(
        self, user_id: str, aggregates: list[Aggregate], conditions: list[Condition]
    ) -> list[float | None]: ...


class CollectionRowSource:
    def __init__(self, info: CollectionInfo):
        self.id = info.id
        self.name = info.name
        self.fields = info.fields
        self.key_field = info.key_field
        self.writable = True

    async def query(self, user_id: str, query: RowQuery) -> RowPage:
        return await collections_db().query_rows(user_id, self.id, query)

    async def aggregate(
        self, user_id: str, aggregates: list[Aggregate], conditions: list[Condition]
    ) -> list[float | None]:
        return await collections_db().aggregate_rows(
            user_id, self.id, aggregates, conditions
        )
