from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pydantic import ValidationError

from backend.data.model import UserOnboarding
from backend.data.onboarding import (
    UserOnboardingUpdate,
    get_user_onboarding,
    reset_user_onboarding,
    update_user_onboarding,
)
from backend.util.json import SafeJson


def wizard_progress(**changes: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "currentStep": "subscription",
        "completedSteps": ["team", "autopilot", "role", "painPoints", "hire"],
        "role": "Founder",
        "otherRole": "",
        "painPoints": ["Research"],
        "otherPainPoint": "",
        "selectedBilling": "monthly",
        "hasUserSelectedBilling": False,
        "selectedCountryCode": "US",
        "hiredTemplateIds": ["template-1"],
        **changes,
    }


def test_wizard_progress_survives_request_validation():
    payload = wizard_progress()
    update = UserOnboardingUpdate.model_validate(
        {"wizardProgress": payload, "wizardRevision": 0, "wizardUserId": "user-one"}
    )

    assert update.model_dump()["wizardProgress"] == payload


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 2},
        {"currentStep": "ONBOARDING_COMPLETE"},
        {"completedSteps": ["ONBOARDING_COMPLETE"]},
        {"completedSteps": ["role"] * 9},
        {"role": "a" * 101},
        {"otherRole": "a" * 101},
        {"painPoints": ["a" * 201]},
        {"painPoints": ["\x00"]},
        {"painPoints": ["Research"] * 21},
        {"otherPainPoint": "a" * 2001},
        {"selectedBilling": "weekly"},
        {"selectedCountryCode": "USA"},
        {"selectedCountryCode": "us"},
        {"hasUserSelectedBilling": "false"},
        {"hiredTemplateIds": ["a" * 129]},
        {"hiredTemplateIds": ["\x00"]},
        {"hiredTemplateIds": ["template"] * 101},
        {"userId": "another-user"},
        {"subscriptionTier": "MAX"},
    ],
)
def test_wizard_progress_rejects_malformed_or_authoritative_fields(changes):
    with pytest.raises(ValidationError):
        UserOnboardingUpdate.model_validate(
            {
                "wizardProgress": wizard_progress(**changes),
                "wizardRevision": 0,
                "wizardUserId": "user-one",
            }
        )


@pytest.fixture
def draft_transaction(mocker):
    tx = Mock()
    tx.useronboarding.update_many = AsyncMock(return_value=1)
    tx.useronboarding.find_unique_or_raise = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=tx)
    context.__aexit__ = AsyncMock(return_value=False)
    mocker.patch("backend.data.onboarding.transaction", return_value=context)
    return tx


@pytest.mark.asyncio(loop_scope="function")
async def test_saving_draft_scopes_user_and_never_completes_or_rewards(
    mocker, draft_transaction
):
    prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    prisma.return_value.upsert = AsyncMock(return_value=Mock(notified=[]))
    reward = mocker.patch("backend.data.onboarding._reward_user", AsyncMock())
    credit = mocker.patch("backend.data.onboarding.get_user_credit_model", AsyncMock())
    notify = mocker.patch(
        "backend.data.onboarding._send_onboarding_notification", AsyncMock()
    )
    completed = mocker.patch("backend.data.onboarding.track_onboarding_completed")
    payload = wizard_progress()

    await update_user_onboarding(
        "authenticated-user",
        UserOnboardingUpdate.model_validate(
            {
                "wizardProgress": payload,
                "wizardRevision": 3,
                "wizardUserId": "authenticated-user",
                "userId": "another-user",
            }
        ),
    )

    calls = prisma.return_value.upsert.call_args_list
    assert all(
        call.kwargs["where"] == {"userId": "authenticated-user"} for call in calls
    )
    assert len(calls) == 1
    draft_transaction.useronboarding.update_many.assert_awaited_once_with(
        where={"userId": "authenticated-user", "wizardRevision": 3},
        data={"wizardProgress": SafeJson(payload), "wizardRevision": {"increment": 1}},
    )
    draft_transaction.useronboarding.find_unique_or_raise.assert_awaited_once_with(
        where={"userId": "authenticated-user"}
    )
    reward.assert_not_awaited()
    credit.assert_not_awaited()
    notify.assert_not_awaited()
    completed.assert_not_called()


@pytest.mark.asyncio(loop_scope="function")
async def test_reading_draft_scopes_each_authenticated_user(mocker):
    prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    prisma.return_value.find_unique = AsyncMock(return_value=None)

    first = await get_user_onboarding("user-one")
    second = await get_user_onboarding("user-two")

    calls = prisma.return_value.find_unique.call_args_list
    assert [call.kwargs["where"] for call in calls] == [
        {"userId": "user-one"},
        {"userId": "user-two"},
    ]
    # Reads create nothing: a user with no row reads as an empty draft at the
    # column's starting revision, which the first save then matches.
    assert (first.userId, first.wizardProgress, first.wizardRevision) == (
        "user-one",
        None,
        0,
    )
    assert second.userId == "user-two"


@pytest.mark.asyncio(loop_scope="function")
async def test_reset_clears_only_the_authenticated_users_draft(mocker):
    prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    prisma.return_value.upsert = AsyncMock()

    await reset_user_onboarding("authenticated-user")

    call = prisma.return_value.upsert.call_args
    assert call.kwargs["where"] == {"userId": "authenticated-user"}
    assert call.kwargs["data"]["update"]["wizardProgress"] is None
    assert call.kwargs["data"]["update"]["wizardRevision"] == {"increment": 1}


@pytest.mark.asyncio(loop_scope="function")
async def test_unrelated_update_does_not_overwrite_draft(mocker):
    prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    prisma.return_value.upsert = AsyncMock(return_value=Mock(notified=[]))

    await update_user_onboarding("user-one", UserOnboardingUpdate(walletShown=True))

    assert prisma.return_value.upsert.call_args.kwargs["data"]["update"] == {
        "walletShown": True
    }


@pytest.mark.asyncio(loop_scope="function")
async def test_explicit_null_clears_draft(mocker, draft_transaction):
    prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    prisma.return_value.upsert = AsyncMock(return_value=Mock(notified=[]))

    await update_user_onboarding(
        "user-one",
        UserOnboardingUpdate.model_validate(
            {"wizardProgress": None, "wizardRevision": 0, "wizardUserId": "user-one"}
        ),
    )

    assert draft_transaction.useronboarding.update_many.call_args.kwargs["data"] == {
        "wizardProgress": None,
        "wizardRevision": {"increment": 1},
    }


def test_response_exposes_nullable_typed_progress():
    schema = UserOnboarding.model_json_schema()

    assert schema["properties"]["wizardProgress"]["anyOf"] == [
        {"$ref": "#/$defs/OnboardingWizardProgress"},
        {"type": "null"},
    ]


@pytest.mark.parametrize("revision", [None, -1, "0", True])
def test_draft_requires_a_valid_revision(revision):
    with pytest.raises(ValidationError):
        UserOnboardingUpdate.model_validate(
            {
                "wizardProgress": wizard_progress(),
                "wizardRevision": revision,
                "wizardUserId": "user-one",
            }
        )


@pytest.mark.asyncio(loop_scope="function")
async def test_stale_writer_cannot_overwrite_newer_draft(mocker, draft_transaction):
    prisma = mocker.patch("backend.data.onboarding.UserOnboarding.prisma")
    prisma.return_value.upsert = AsyncMock(return_value=Mock(notified=[]))
    tx = draft_transaction
    tx.useronboarding.update_many = AsyncMock(return_value=0)

    with pytest.raises(RuntimeError, match="Onboarding progress changed"):
        await update_user_onboarding(
            "user-one",
            UserOnboardingUpdate.model_validate(
                {
                    "wizardProgress": wizard_progress(),
                    "wizardRevision": 3,
                    "wizardUserId": "user-one",
                }
            ),
        )

    tx.useronboarding.update_many.assert_awaited_once()
    tx.useronboarding.find_unique_or_raise.assert_not_awaited()
    assert prisma.return_value.upsert.await_count == 1


@pytest.mark.asyncio(loop_scope="function")
async def test_cookie_account_switch_cannot_save_previous_users_draft(
    mocker, draft_transaction
):
    read = mocker.patch(
        "backend.data.onboarding.get_user_onboarding",
        AsyncMock(return_value=Mock(notified=[])),
    )
    update = UserOnboardingUpdate.model_validate(
        {
            "wizardProgress": wizard_progress(),
            "wizardRevision": 0,
            "wizardUserId": "previous-user",
        }
    )

    with pytest.raises(RuntimeError, match="Onboarding progress changed"):
        await update_user_onboarding("authenticated-user", update)

    read.assert_not_awaited()
    draft_transaction.useronboarding.update_many.assert_not_awaited()


@pytest.mark.parametrize("owner", [None, "", 42, "a" * 129])
def test_draft_requires_a_valid_captured_user(owner):
    with pytest.raises(ValidationError):
        UserOnboardingUpdate.model_validate(
            {
                "wizardProgress": wizard_progress(),
                "wizardRevision": 0,
                "wizardUserId": owner,
            }
        )
