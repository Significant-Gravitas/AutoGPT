"""Builder search history against a real Postgres (``server`` starts the stack)."""

import uuid

import prisma.models
import pytest

from backend.api.features.builder import db
from backend.api.features.builder.model import SearchEntry
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_a_user_updates_their_own_entry_in_place(server: SpinTestServer):
    [owner] = await _make_users(1)
    try:
        search_id = await db.update_search(owner, SearchEntry(search_query="alpha"))

        returned = await db.update_search(
            owner, SearchEntry(search_query="beta", search_id=search_id)
        )

        assert returned == search_id
        rows = await prisma.models.BuilderSearchHistory.prisma().find_many(
            where={"userId": owner}
        )
        assert [(r.id, r.searchQuery) for r in rows] == [(search_id, "beta")]
    finally:
        await _drop_users([owner])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("target", ["another user's entry", "an unknown id"])
async def test_an_id_that_is_not_the_users_creates_their_own_entry(
    server: SpinTestServer, target: str
):
    owner, caller = await _make_users(2)
    try:
        owners_id = await db.update_search(owner, SearchEntry(search_query="alpha"))
        search_id = owners_id if target == "another user's entry" else str(uuid.uuid4())

        returned = await db.update_search(
            caller, SearchEntry(search_query="beta", search_id=search_id)
        )

        assert returned not in (owners_id, search_id)
        owners_row = await prisma.models.BuilderSearchHistory.prisma().find_unique(
            where={"id": owners_id}
        )
        assert owners_row is not None
        assert (owners_row.userId, owners_row.searchQuery) == (owner, "alpha")
        callers_row = await prisma.models.BuilderSearchHistory.prisma().find_unique(
            where={"id": returned}
        )
        assert callers_row is not None
        assert (callers_row.userId, callers_row.searchQuery) == (caller, "beta")
    finally:
        await _drop_users([owner, caller])


async def _make_users(count: int) -> list[str]:
    user_ids = [f"search-history-{uuid.uuid4().hex[:16]}" for _ in range(count)]
    await prisma.models.User.prisma().create_many(
        data=[{"id": uid, "email": f"{uid}@example.invalid"} for uid in user_ids]
    )
    return user_ids


async def _drop_users(user_ids: list[str]) -> None:
    await prisma.models.User.prisma().delete_many(where={"id": {"in": user_ids}})
