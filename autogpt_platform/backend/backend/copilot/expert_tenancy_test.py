"""A turn shares the expert tenancy it checked with itself, and nothing else."""

import asyncio

import pytest

from backend.copilot.expert_tenancy import (
    checked_expert_tenancy,
    remember_expert_tenancy,
    sharing_expert_tenancy,
)

TENANCY = ("personal-org", "personal-team")


def test_a_turn_reuses_the_tenancy_it_checked_for_that_expert_only():
    with sharing_expert_tenancy():
        remember_expert_tenancy("user-1", "expert-1", TENANCY)

        assert checked_expert_tenancy("user-1", "expert-1") == TENANCY
        assert checked_expert_tenancy("user-1", "expert-2") is None
        assert checked_expert_tenancy("user-2", "expert-1") is None


def test_the_next_turn_checks_again():
    with sharing_expert_tenancy():
        remember_expert_tenancy("user-1", "expert-1", TENANCY)
    with sharing_expert_tenancy():
        assert checked_expert_tenancy("user-1", "expert-1") is None


def test_nothing_is_remembered_outside_a_turn():
    remember_expert_tenancy("user-1", "expert-1", TENANCY)

    assert checked_expert_tenancy("user-1", "expert-1") is None


@pytest.mark.asyncio
async def test_tasks_the_turn_started_see_what_it_checked_after_they_started():
    """The engine's stream runs in a task the turn starts before it checks
    the expert, so the check has to reach a task that already exists."""
    checked = asyncio.Event()
    seen = []

    async def engine() -> None:
        await checked.wait()
        seen.append(checked_expert_tenancy("user-1", "expert-1"))

    with sharing_expert_tenancy():
        task = asyncio.create_task(engine())
        remember_expert_tenancy("user-1", "expert-1", TENANCY)
        checked.set()
        await task

    assert seen == [TENANCY]
