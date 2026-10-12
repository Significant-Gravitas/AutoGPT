"""DB-backed reads over delegated sub-sessions."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import Expert, PlatformCostLog, User

from backend.data.db import prisma as db_client
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

from .delegation_db import (
    get_delegation_settings,
    get_delegation_spend_since,
    get_expert_hired_at,
    get_session_costs,
    raise_delegation_cap,
    stop_delegation_at_cap,
    update_delegation_settings,
)
from .delegation_settings import DelegationSettings


@pytest_asyncio.fixture(autouse=True)
async def absorb_a_stale_event_loop(server: SpinTestServer):
    """An earlier test can leave the shared Prisma client bound to a loop that
    has since closed; only the first query on the new loop fails, and the engine
    re-establishes itself. Spend that failure here rather than in a fixture
    (same guard as ``experts_db_test``)."""
    try:
        await db_client.execute_raw("SELECT 1")
    except RuntimeError as error:
        if "Event loop is closed" not in str(error):
            raise


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


async def _session(
    user_id: str, metadata: dict | None = None, expert_id: str | None = None
) -> str:
    row = await PrismaChatSession.prisma().create(
        data={
            "userId": user_id,
            "credentials": SafeJson({}),
            "metadata": SafeJson(metadata or {}),
            **({"expertId": expert_id} if expert_id else {}),
        }
    )
    return row.id


async def _cost(
    user_id: str, session_id: str, microdollars: int, at: datetime | None = None
) -> None:
    await PlatformCostLog.prisma().create(
        data={
            "userId": user_id,
            "chatSessionId": session_id,
            "provider": "anthropic",
            "costMicrodollars": microdollars,
            **({"createdAt": at} if at else {}),
        }
    )


async def _expert(user_id: str) -> str:
    row = await Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": f"Bea {uuid.uuid4().hex[:6]}",
            "role": "PM",
            "identity": "",
        }
    )
    return row.id


@pytest.fixture
async def users():
    created = [await _user(), await _user()]
    yield created
    for user_id in created:
        await PlatformCostLog.prisma().delete_many(where={"userId": user_id})
        await PrismaChatSession.prisma().delete_many(where={"userId": user_id})
        await Expert.prisma().delete_many(where={"ownerUserId": user_id})
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


@pytest.mark.asyncio(loop_scope="session")
async def test_delegation_spend_counts_only_delegated_threads_since_then(users):
    alice, bob = users
    since = datetime.now(UTC) - timedelta(hours=1)
    otto = await _session(alice)
    bea = await _expert(alice)
    delegated = await _session(alice, {"delegated_by_session_id": otto}, expert_id=bea)
    # A run_sub_session sub carries the same provenance but stays in scope.
    isolate = await _session(alice, {"delegated_by_session_id": otto})
    await _cost(alice, delegated, 300_000)
    await _cost(alice, delegated, 50_000, at=since - timedelta(minutes=5))
    await _cost(alice, isolate, 700_000)
    await _cost(alice, otto, 900_000)
    await _cost(bob, delegated, 4_000_000)

    assert await get_delegation_spend_since(alice, since) == 300_000


@pytest.mark.asyncio(loop_scope="session")
async def test_settings_default_until_saved_then_round_trip(users):
    alice, bob = users

    assert await get_delegation_settings(alice) == DelegationSettings()

    saved = DelegationSettings(mode="ask_first", per_delegation_cap_usd=5.0)
    assert await update_delegation_settings(alice, saved) == saved
    assert await get_delegation_settings(alice) == saved
    assert await get_delegation_settings(bob) == DelegationSettings()


@pytest.mark.asyncio(loop_scope="session")
async def test_hire_date_is_only_the_owners_to_read(users):
    alice, bob = users
    bea = await _expert(alice)

    hired = await get_expert_hired_at(alice, bea)

    assert hired is not None
    assert datetime.now(UTC) - hired < timedelta(minutes=5)
    assert await get_expert_hired_at(bob, bea) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_the_cap_is_raised_and_stopped_for_the_owner_only(users):
    alice, bob = users
    thread = await _session(alice, {"delegation_cap_usd": 2.0, "purpose": "keep"})

    assert await raise_delegation_cap(thread, alice, 5.0, "q1") == 7.0
    # A retried answer to the same question raises nothing twice.
    assert await raise_delegation_cap(thread, alice, 5.0, "q1") == 7.0
    assert await raise_delegation_cap(thread, bob, 100.0, "q2") == 0.0
    await stop_delegation_at_cap(thread, bob)
    row = await PrismaChatSession.prisma().find_unique(where={"id": thread})
    assert row is not None and row.metadata == {
        "delegation_cap_usd": 7.0,
        "delegation_cap_raised_for": "q1",
        "purpose": "keep",
    }
    assert await raise_delegation_cap(thread, alice, 1.0, "q2") == 8.0

    await stop_delegation_at_cap(thread, alice)
    row = await PrismaChatSession.prisma().find_unique(where={"id": thread})
    assert row is not None and row.metadata["delegation_cap_stopped"] is True
