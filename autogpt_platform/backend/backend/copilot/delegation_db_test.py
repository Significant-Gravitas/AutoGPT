"""DB-backed reads over delegated sub-sessions."""

import uuid

import pytest
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import PlatformCostLog, User

from backend.util.json import SafeJson

from .delegation_db import get_session_costs


async def _user() -> str:
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"delegation-{user_id}@example.com",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
        }
    )
    return user_id


async def _session(user_id: str, metadata: dict | None = None) -> str:
    row = await PrismaChatSession.prisma().create(
        data={
            "userId": user_id,
            "credentials": SafeJson({}),
            "metadata": SafeJson(metadata or {}),
        }
    )
    return row.id


async def _cost(user_id: str, session_id: str, microdollars: int) -> None:
    await PlatformCostLog.prisma().create(
        data={
            "userId": user_id,
            "chatSessionId": session_id,
            "provider": "anthropic",
            "costMicrodollars": microdollars,
        }
    )


@pytest.fixture
async def users():
    created = [await _user(), await _user()]
    yield created
    for user_id in created:
        await PlatformCostLog.prisma().delete_many(where={"userId": user_id})
        await PrismaChatSession.prisma().delete_many(where={"userId": user_id})
        await User.prisma().delete(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_session_costs_sum_per_session_for_the_owner_only(users):
    alice, bob = users
    first, second = await _session(alice), await _session(alice)
    await _cost(alice, first, 100_000)
    await _cost(alice, first, 20_000)
    await _cost(alice, second, 5_000)
    # Another user's row naming the same chat never counts toward it.
    await _cost(bob, first, 9_000_000)

    costs = await get_session_costs(alice, [first, second, "missing"])

    assert costs == {first: 120_000, second: 5_000}
