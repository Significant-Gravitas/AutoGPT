"""Saving and clearing the onboarding wizard's draft, against the real database.

Needs the database (``server`` starts the test stack). A mocked Prisma client
accepts payloads the real one refuses, such as a bare ``None`` for a Json column,
and can't show which saves a conditional write refuses.
"""

import uuid

import pytest
from prisma.models import User, UserOnboarding

from backend.data.onboarding import (
    UserOnboardingUpdate,
    reset_user_onboarding,
    update_user_onboarding,
)
from backend.data.onboarding_wizard import OnboardingWizardConflict
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
                "role": "Marketing",
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
    # The role is kept from the draft, so it is reset with it.
    assert (reset.role, other.role) == (None, "Marketing")


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


async def test_a_draft_past_the_role_step_keeps_its_role(users_with_drafts):
    user_id, _ = users_with_drafts
    draft = wizard_progress(role="Other", otherRole="Dentist")

    await update_user_onboarding(
        user_id,
        UserOnboardingUpdate.model_validate(
            {
                "wizardProgress": draft,
                "wizardRevision": DRAFT_REVISION,
                "wizardUserId": user_id,
            }
        ),
    )

    row = await read_row(user_id)
    assert (row.role, row.roleOther, row.wizardRevision) == (
        "Other",
        "Dentist",
        DRAFT_REVISION + 1,
    )


async def test_a_tabs_next_save_is_kept_over_its_own_unanswered_one(
    users_with_drafts,
):
    """The tab's first save landed but never answered, so its next save still
    carries the revision from before the first (#17158)."""
    user_id, _ = users_with_drafts
    await save(user_id, DRAFT_REVISION, tab="tab-1", number=1, role="Founder")

    saved = await save(user_id, DRAFT_REVISION, tab="tab-1", number=2, role="Sales")

    row = await read_row(user_id)
    assert (row.wizardProgress, row.wizardRevision, saved.wizardRevision) == (
        wizard_progress(role="Sales"),
        DRAFT_REVISION + 2,
        DRAFT_REVISION + 2,
    )


async def test_the_same_save_arriving_twice_is_not_refused(users_with_drafts):
    user_id, _ = users_with_drafts
    await save(user_id, DRAFT_REVISION, tab="tab-1", number=1, role="Founder")

    await save(user_id, DRAFT_REVISION, tab="tab-1", number=1, role="Founder")

    row = await read_row(user_id)
    assert (row.wizardProgress, row.wizardRevision) == (
        wizard_progress(role="Founder"),
        DRAFT_REVISION + 2,
    )


async def test_an_older_save_landing_late_cannot_replace_a_newer_one(
    users_with_drafts,
):
    user_id, _ = users_with_drafts
    await save(user_id, DRAFT_REVISION, tab="tab-1", number=2, role="Sales")

    with pytest.raises(OnboardingWizardConflict):
        await save(user_id, DRAFT_REVISION, tab="tab-1", number=1, role="Founder")

    row = await read_row(user_id)
    assert (row.wizardProgress, row.wizardRevision) == (
        wizard_progress(role="Sales"),
        DRAFT_REVISION + 1,
    )


async def test_another_tab_saving_from_the_same_revision_is_refused(
    users_with_drafts,
):
    user_id, _ = users_with_drafts
    await save(user_id, DRAFT_REVISION, tab="tab-1", number=1, role="Founder")

    with pytest.raises(OnboardingWizardConflict):
        await save(user_id, DRAFT_REVISION, tab="tab-2", number=1, role="Sales")

    row = await read_row(user_id)
    assert row.wizardProgress == wizard_progress(role="Founder")


@pytest.mark.parametrize("saved_between", ["another tab", "no tab", "reset"])
async def test_a_tabs_stale_save_is_refused_once_anything_else_saved(
    users_with_drafts, saved_between
):
    """Only the tab's own saves may sit between the revision it read and its
    next save; another tab's, an older client's that names no tab, or a reset
    leaves it a real conflict."""
    user_id, _ = users_with_drafts
    await save(user_id, DRAFT_REVISION, tab="tab-1", number=1, role="Founder")
    if saved_between == "another tab":
        await save(user_id, DRAFT_REVISION + 1, tab="tab-2", number=1, role="Sales")
    elif saved_between == "no tab":
        await save(user_id, DRAFT_REVISION + 1, role="Sales")
    else:
        await reset_user_onboarding(user_id)
    before = await read_row(user_id)

    with pytest.raises(OnboardingWizardConflict):
        await save(user_id, DRAFT_REVISION, tab="tab-1", number=2, role="Marketing")

    row = await read_row(user_id)
    assert (row.wizardProgress, row.wizardRevision) == (
        before.wizardProgress,
        DRAFT_REVISION + 2,
    )


async def save(
    user_id: str,
    revision: int,
    *,
    tab: str | None = None,
    number: int | None = None,
    **changes: str,
):
    request = {
        "wizardProgress": wizard_progress(**changes),
        "wizardRevision": revision,
        "wizardUserId": user_id,
    }
    if tab is not None:
        request |= {"wizardWriterId": tab, "wizardWriteSeq": number}
    return await update_user_onboarding(
        user_id, UserOnboardingUpdate.model_validate(request)
    )


async def read_row(user_id: str) -> UserOnboarding:
    return await UserOnboarding.prisma().find_unique_or_raise(where={"userId": user_id})
