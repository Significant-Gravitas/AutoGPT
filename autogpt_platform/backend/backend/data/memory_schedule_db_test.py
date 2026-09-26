"""The memory-scope schedule registry against a real Postgres: concurrent
first claims of one scope, and a pause racing them.

Needs the database (``server`` starts the test stack). The unit tests in
``memory_schedule_test.py`` cover the same code with Prisma mocked.
"""

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest
from prisma.enums import MemoryScopeScheduleState

from backend.copilot.dream import registry
from backend.copilot.graphiti.scope import MemoryScope
from backend.data import memory_schedule
from backend.data.memory_schedule import MemoryScopeSchedule
from backend.util.test import SpinTestServer

SCOPES = 10
CONTENDERS = 20


async def _make_users(prefix: str) -> list[str]:
    user_ids = [f"{prefix}-{uuid.uuid4().hex[:16]}" for _ in range(SCOPES)]
    await prisma.models.User.prisma().create_many(
        data=[{"id": uid, "email": f"{uid}@example.invalid"} for uid in user_ids]
    )
    return user_ids


async def _drop_users(user_ids: list[str]) -> None:
    await prisma.models.User.prisma().delete_many(where={"id": {"in": user_ids}})


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_first_claims_all_return_the_one_row(server: SpinTestServer):
    user_ids = await _make_users("claim-race")
    try:
        for user_id in user_ids:
            results = await asyncio.gather(
                *(
                    memory_schedule.claim_scope_schedule(user_id, user_id, None, "UTC")
                    for _ in range(CONTENDERS)
                ),
                return_exceptions=True,
            )
            failures = [r for r in results if not isinstance(r, MemoryScopeSchedule)]
            assert failures == []
            rows = await prisma.models.MemoryScopeSchedule.prisma().count(
                where={"userId": user_id}
            )
            assert rows == 1
    finally:
        await _drop_users(user_ids)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pause_racing_first_registrations_always_ends_paused(
    server: SpinTestServer,
):
    """A pause of a scope with no row claims it PAUSED while registrations
    claim it ACTIVE. In every other round a registration's claim lands
    first, so the pause's own claim loses and gets the ACTIVE row back;
    either way the pause ends PAUSED with the crons removed."""
    user_ids = await _make_users("pause-race")
    try:
        for round_number, user_id in enumerate(user_ids):
            await _race_pause_against_claims(
                user_id, registration_first=round_number % 2 == 1
            )
    finally:
        await _drop_users(user_ids)


async def _race_pause_against_claims(user_id: str, registration_first: bool) -> None:
    barrier = asyncio.Barrier(CONTENDERS + 1)
    registered = asyncio.Event()

    async def register(*args: Any) -> MemoryScopeSchedule:
        await barrier.wait()
        row = await memory_schedule.claim_scope_schedule(*args)
        registered.set()
        return row

    async def pause_claim(*args: Any) -> MemoryScopeSchedule:
        await barrier.wait()
        if registration_first:
            await registered.wait()
        return await memory_schedule.claim_scope_schedule(*args)

    db = SimpleNamespace(
        set_scope_state=memory_schedule.set_scope_state,
        claim_scope_schedule=pause_claim,
    )
    remove = AsyncMock(return_value=True)
    with (
        patch.object(registry, "memory_schedule_db", return_value=db),
        patch.object(registry, "resolve_user_timezone", AsyncMock(return_value="UTC")),
        patch.object(registry, "remove_scope_jobs", remove),
    ):
        paused, *_ = await asyncio.gather(
            registry.pause_scope(MemoryScope.for_user(user_id)),
            *(register(user_id, user_id, None, "UTC") for _ in range(CONTENDERS)),
        )
    row = await memory_schedule.get_scope_schedule(user_id, user_id)
    assert paused is True
    assert row is not None and row.state == MemoryScopeScheduleState.PAUSED
    remove.assert_awaited_once()
