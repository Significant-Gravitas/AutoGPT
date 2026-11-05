"""How a turn's expert check reads: no database, so none of experts_db_test's
seeding or cleanup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest

from backend.api.features.experts import experts_db


def _reading_once_both_started(started: dict[str, asyncio.Event], own: str, result):
    """A read that returns *result* only once the other read has started too."""

    async def read(*args, **kwargs):
        started[own].set()
        await asyncio.wait_for(asyncio.gather(*(e.wait() for e in started.values())), 2)
        return result

    return read


@pytest.mark.asyncio(loop_scope="session")
async def test_check_private_expert_reads_the_hire_beside_its_owners_scope():
    """The expert's row and its owner's scope don't depend on each other, so
    a turn's check reads them side by side rather than one after the other."""
    started = {"row": asyncio.Event(), "scope": asyncio.Event()}
    row = SimpleNamespace(id="expert-1")
    manager = SimpleNamespace(
        find_first=AsyncMock(
            side_effect=_reading_once_both_started(started, "row", row)
        )
    )
    owner_scope = AsyncMock(
        side_effect=_reading_once_both_started(
            started, "scope", ("personal-org", "personal-team")
        )
    )
    with (
        patch.object(prisma.models.Expert, "prisma", return_value=manager),
        patch.object(experts_db, "get_user_default_team", new=owner_scope),
        patch.object(experts_db, "_to_model", return_value=row),
        patch.object(experts_db, "CheckedExpert", side_effect=SimpleNamespace),
    ):
        check = await experts_db.check_private_expert("owner-1", "expert-1")

    assert check.expert is row
    assert (check.organization_id, check.team_id) == ("personal-org", "personal-team")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_missing_hire_is_the_error_when_its_owner_has_no_scope_either():
    """The scope's failure comes back first here, but a turn for an expert that
    is gone must say so, not that its tenancy is temporarily missing."""
    expert_lookup = asyncio.Event()

    async def no_row(*args, **kwargs):
        await asyncio.wait_for(expert_lookup.wait(), 2)
        return None

    async def no_scope(user_id: str):
        expert_lookup.set()
        return None, None

    with (
        patch.object(
            prisma.models.Expert,
            "prisma",
            return_value=SimpleNamespace(find_first=AsyncMock(side_effect=no_row)),
        ),
        patch.object(
            experts_db, "get_user_default_team", new=AsyncMock(side_effect=no_scope)
        ),
        pytest.raises(experts_db.ExpertNotFoundError),
    ):
        await experts_db.check_private_expert("owner-1", "expert-1")


@pytest.mark.asyncio(loop_scope="session")
async def test_a_missing_hire_leaves_its_owners_scope_read_finished():
    """On a user's first touch the scope read creates their personal
    organization, so a missing expert waits for it instead of cancelling it
    midway or leaving it running after the check has answered."""
    finished: list[bool] = []
    scope_started = asyncio.Event()

    async def no_row(*args, **kwargs):
        await asyncio.wait_for(scope_started.wait(), 2)
        return None

    async def bootstrapping_scope(user_id: str):
        scope_started.set()
        await asyncio.sleep(0.05)
        finished.append(True)
        return "personal-org", "personal-team"

    with (
        patch.object(
            prisma.models.Expert,
            "prisma",
            return_value=SimpleNamespace(find_first=AsyncMock(side_effect=no_row)),
        ),
        patch.object(
            experts_db,
            "get_user_default_team",
            new=AsyncMock(side_effect=bootstrapping_scope),
        ),
        pytest.raises(experts_db.ExpertNotFoundError),
    ):
        await experts_db.check_private_expert("owner-1", "expert-1")

    assert finished == [True]
