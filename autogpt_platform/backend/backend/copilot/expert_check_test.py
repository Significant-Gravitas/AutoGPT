"""A turn shares the expert check it made with itself, and nothing else."""

import asyncio

import pytest

from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    CheckedExpert,
    Expert,
)
from backend.copilot.expert_check import (
    checked_expert,
    forget_expert_checks,
    remember_expert_check,
    sharing_expert_checks,
)

CHECK = CheckedExpert(
    expert=Expert(
        id="expert-1",
        name="Maria",
        avatar_url=None,
        role="SEO Specialist",
        tagline=None,
        bio=None,
        skills=[],
        identity="You are Maria.",
        voice_preferences="Direct.",
        boundaries="Ask first.",
        protected_soul_rules=list(PROTECTED_SOUL_RULES),
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    ),
    organization_id="personal-org",
    team_id="personal-team",
)


def test_a_turn_reuses_the_check_it_made_for_that_expert_only():
    with sharing_expert_checks():
        remember_expert_check("user-1", "expert-1", CHECK)

        assert checked_expert("user-1", "expert-1") == CHECK
        assert checked_expert("user-1", "expert-2") is None
        assert checked_expert("user-2", "expert-1") is None


def test_the_next_turn_checks_again():
    with sharing_expert_checks():
        remember_expert_check("user-1", "expert-1", CHECK)
    with sharing_expert_checks():
        assert checked_expert("user-1", "expert-1") is None


def test_a_forgotten_check_is_read_again():
    with sharing_expert_checks():
        remember_expert_check("user-1", "expert-1", CHECK)
        forget_expert_checks()

        assert checked_expert("user-1", "expert-1") is None


def test_nothing_is_remembered_outside_a_turn():
    remember_expert_check("user-1", "expert-1", CHECK)

    assert checked_expert("user-1", "expert-1") is None


@pytest.mark.asyncio
async def test_tasks_the_turn_started_see_what_it_checked_after_they_started():
    """The engine's stream runs in a task the turn starts before it checks
    the expert, so the check has to reach a task that already exists."""
    checked = asyncio.Event()
    seen = []

    async def engine() -> None:
        await checked.wait()
        seen.append(checked_expert("user-1", "expert-1"))

    with sharing_expert_checks():
        task = asyncio.create_task(engine())
        remember_expert_check("user-1", "expert-1", CHECK)
        checked.set()
        await task

    assert seen == [CHECK]
