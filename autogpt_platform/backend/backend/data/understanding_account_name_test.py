"""AutoPilot's understanding reads and writes the user's name on the account
(`User.name`); its row keeps its own copy only as graph memory's label."""

import uuid
from datetime import UTC, datetime

import pytest
from prisma.models import CoPilotUnderstanding, User

from backend.copilot.graphiti.ingest import resolve_user_name
from backend.data.redis_client import get_redis_async
from backend.data.understanding import (
    BusinessUnderstanding,
    BusinessUnderstandingInput,
    _cache_key,
    format_understanding_for_prompt,
    get_business_understanding,
    upsert_business_understanding,
)
from backend.data.user import get_user_by_id
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_otto_reads_the_account_name_over_the_understandings_old_copy(
    server: SpinTestServer,
):
    user_id = await _account("Sam")
    redis = await get_redis_async()
    try:
        # A row and a cached copy written while the understanding kept its own name.
        await CoPilotUnderstanding.prisma().create(
            data={
                "userId": user_id,
                "data": SafeJson(
                    {"name": "jane.doe", "business": {"industry": "Retail"}}
                ),
            }
        )
        now = datetime.now(UTC)
        stale = BusinessUnderstanding(
            id="old",
            user_id=user_id,
            created_at=now,
            updated_at=now,
            user_name="jane.doe",
            industry="Retail",
        )
        await redis.set(_cache_key(user_id), stale.model_dump_json(), ex=60)

        cached = await get_business_understanding(user_id)
        await redis.delete(_cache_key(user_id))
        stored = await get_business_understanding(user_id)

        for understanding in (cached, stored):
            assert understanding is not None
            context = format_understanding_for_prompt(understanding)
            assert "Name: Sam" in context
            assert "jane.doe" not in context
    finally:
        await _remove([user_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_a_name_given_to_the_understanding_fills_only_an_empty_account_name(
    server: SpinTestServer,
):
    named, no_name, empty_name = [await _account(n) for n in ("Sam", None, "")]
    try:
        # Read once, so the write must replace a cached account without a name.
        await get_user_by_id(no_name)
        given = {}
        for user_id in (named, no_name, empty_name):
            given[user_id] = await upsert_business_understanding(
                user_id, BusinessUnderstandingInput(user_name="Rob", industry="Retail")
            )

        accounts = {
            user.id: user.name
            for user in await User.prisma().find_many(
                where={"id": {"in": [named, no_name, empty_name]}}
            )
        }
        assert accounts == {named: "Sam", no_name: "Rob", empty_name: "Rob"}
        assert {uid: u.user_name for uid, u in given.items()} == accounts
        rows = await CoPilotUnderstanding.prisma().find_many(
            where={"userId": {"in": [named, no_name, empty_name]}}
        )
        assert [row.data for row in rows] == [
            {"name": "Rob", "business": {"industry": "Retail", "version": 1}}
        ] * 3
    finally:
        await _remove([named, no_name, empty_name])


@pytest.mark.asyncio(loop_scope="session")
async def test_memory_labels_the_user_by_the_understandings_name_not_the_account(
    server: SpinTestServer,
):
    user_id = await _account("Sam")
    try:
        await CoPilotUnderstanding.prisma().create(
            data={"userId": user_id, "data": SafeJson({"name": "jane.doe"})}
        )
        assert await resolve_user_name(user_id) == "jane.doe"

        await upsert_business_understanding(
            user_id, BusinessUnderstandingInput(industry="Retail")
        )
        assert await resolve_user_name(user_id) == "jane.doe"

        await upsert_business_understanding(
            user_id, BusinessUnderstandingInput(user_name="Rob")
        )
        assert await resolve_user_name(user_id) == "Rob"
        assert (await get_user_by_id(user_id)).name == "Sam"
    finally:
        await _remove([user_id])


async def _account(name: str | None) -> str:
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.com", "name": name}
    )
    return user_id


async def _remove(user_ids: list[str]) -> None:
    redis = await get_redis_async()
    for user_id in user_ids:
        await redis.delete(_cache_key(user_id))
        get_user_by_id.cache_delete(user_id)
    await User.prisma().delete_many(where={"id": {"in": user_ids}})
