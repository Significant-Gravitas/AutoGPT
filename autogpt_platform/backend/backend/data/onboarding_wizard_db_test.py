"""Clearing the onboarding wizard's saved draft, against the real database.

Needs the database (``server`` starts the test stack). A mocked Prisma client
accepts payloads the real one refuses, such as a bare ``None`` for a Json column.
"""

import uuid

import pytest
from prisma.models import User, UserOnboarding

from backend.data.onboarding import (
    UserOnboardingUpdate,
    reset_user_onboarding,
    update_user_onboarding,
)
from backend.data.onboarding_wizard_test import wizard_progress
from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

DRAFT_REVISION = 3


@pytest.fixture
async def users_with_drafts(server: SpinTestServer):
    user_ids = [f"onboarding-draft-{uuid.uuid4()}" for _ in range(2)]
    for user_id in user_ids:
        await User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.com"}
        )
        await UserOnboarding.prisma().create(
            data={
                "userId": user_id,
                "wizardProgress": SafeJson(wizard_progress()),
                "wizardRevision": DRAFT_REVISION,
            }
        )
    yield user_ids
    await User.prisma().delete_many(where={"id": {"in": user_ids}})


async def test_reset_clears_only_the_users_own_draft(users_with_drafts):
    user_id, other_user_id = users_with_drafts

    await reset_user_onboarding(user_id)

    reset = await read_row(user_id)
    other = await read_row(other_user_id)
    assert (reset.wizardProgress, reset.wizardRevision) == (None, DRAFT_REVISION + 1)
    assert (other.wizardProgress, other.wizardRevision) == (
        wizard_progress(),
        DRAFT_REVISION,
    )


async def test_explicit_null_clears_the_draft(users_with_drafts):
    user_id, _ = users_with_drafts

    await update_user_onboarding(
        user_id,
        UserOnboardingUpdate.model_validate(
            {
                "wizardProgress": None,
                "wizardRevision": DRAFT_REVISION,
                "wizardUserId": user_id,
            }
        ),
    )

    row = await read_row(user_id)
    assert (row.wizardProgress, row.wizardRevision) == (None, DRAFT_REVISION + 1)


async def read_row(user_id: str) -> UserOnboarding:
    return await UserOnboarding.prisma().find_unique_or_raise(where={"userId": user_id})
