from unittest.mock import AsyncMock, MagicMock

import pytest
from autogpt_libs.auth import get_user_id, requires_user
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.features.onboarding import routes
from backend.api.features.onboarding.routes import router
from backend.data.onboarding import _default_user_onboarding
from backend.data.onboarding_role import OnboardingRole
from backend.data.onboarding_wizard import OnboardingWizardConflict


async def authenticated_user() -> str:
    return "authenticated-user"


async def authenticated() -> None:
    return None


@pytest.mark.asyncio
async def test_draft_conflict_returns_409_for_authenticated_user(mocker):
    update = mocker.patch(
        "backend.api.features.onboarding.routes.update_user_onboarding",
        AsyncMock(side_effect=OnboardingWizardConflict()),
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[requires_user] = authenticated
    app.dependency_overrides[get_user_id] = authenticated_user

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.patch(
            "/onboarding",
            json={
                "wizardProgress": None,
                "wizardRevision": 3,
                "wizardUserId": "authenticated-user",
                "userId": "another-user",
            },
        )

    assert response.status_code == 409
    assert "changed in another session" in response.json()["detail"]
    assert update.await_args.args[0] == "authenticated-user"


@pytest.mark.asyncio
async def test_draft_without_revision_returns_422_without_writing(mocker):
    update = mocker.patch(
        "backend.api.features.onboarding.routes.update_user_onboarding", AsyncMock()
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[requires_user] = authenticated
    app.dependency_overrides[get_user_id] = authenticated_user

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.patch(
            "/onboarding",
            json={"wizardProgress": None, "wizardUserId": "authenticated-user"},
        )

    assert response.status_code == 422
    update.assert_not_awaited()


USER_ID = "authenticated-user"


def _draft(**fields) -> dict:
    return {
        "version": 1,
        "currentStep": "painPoints",
        "completedSteps": ["team", "autopilot", "role"],
        "role": "Marketing",
        "otherRole": "",
        "painPoints": [],
        "otherPainPoint": "",
        "selectedBilling": "monthly",
        "hasUserSelectedBilling": False,
        "selectedCountryCode": "US",
        "hiredTemplateIds": [],
        **fields,
    }


def _saved_row(kept: OnboardingRole | None):
    """The row a draft save returns: it holds the pick kept before it."""
    return _default_user_onboarding(USER_ID).model_copy(
        update={
            "role": kept.choice if kept else None,
            "roleOther": kept.other if kept else None,
        }
    )


@pytest.fixture
def role_writes(mocker):
    """The kept pick's copy, and its MailerLite and PostHog writes."""
    return {
        "save": mocker.patch.object(routes, "save_onboarding_role", new=AsyncMock()),
        "queue": mocker.patch.object(routes, "queue_onboarding_role", new=AsyncMock()),
        "posthog": mocker.patch.object(routes, "set_onboarding_role", new=MagicMock()),
    }


async def _save_draft(mocker, draft: dict, kept: OnboardingRole | None = None):
    mocker.patch.object(
        routes, "update_user_onboarding", AsyncMock(return_value=_saved_row(kept))
    )
    return await _patch(draft)


async def _patch(draft: dict):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[requires_user] = authenticated
    app.dependency_overrides[get_user_id] = authenticated_user
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.patch(
            "/onboarding",
            json={
                "wizardProgress": draft,
                "wizardRevision": 3,
                "wizardUserId": USER_ID,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields, role",
    [
        ({}, OnboardingRole(choice="Marketing")),
        (
            {"role": "Other", "otherRole": "  Dentist "},
            OnboardingRole(choice="Other", other="Dentist"),
        ),
        # The saves of later steps carry the pick too, so one a lost write
        # missed is kept by the next.
        (
            {
                "currentStep": "subscription",
                "completedSteps": ["role", "painPoints", "hire"],
            },
            OnboardingRole(choice="Marketing"),
        ),
    ],
    ids=["option", "other", "at-the-paywall"],
)
async def test_moving_on_from_the_role_step_keeps_and_sends_the_pick(
    mocker, role_writes, fields, role
):
    response = await _save_draft(mocker, _draft(**fields))

    assert response.status_code == 200
    role_writes["save"].assert_awaited_once_with(USER_ID, role)
    role_writes["queue"].assert_awaited_once_with(USER_ID, role)
    role_writes["posthog"].assert_called_once_with(user_id=USER_ID, role=role)


@pytest.mark.asyncio
async def test_a_changed_pick_replaces_the_kept_one(mocker, role_writes):
    response = await _save_draft(
        mocker, _draft(role="Engineering"), kept=OnboardingRole(choice="Marketing")
    )

    assert response.status_code == 200
    engineering = OnboardingRole(choice="Engineering")
    role_writes["save"].assert_awaited_once_with(USER_ID, engineering)
    role_writes["queue"].assert_awaited_once_with(USER_ID, engineering)
    role_writes["posthog"].assert_called_once_with(user_id=USER_ID, role=engineering)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields, kept",
    [
        # The kept pick: every later step's save is a no-op.
        ({}, OnboardingRole(choice="Marketing")),
        (
            {"role": "Other", "otherRole": "Dentist"},
            OnboardingRole(choice="Other", other="Dentist"),
        ),
        # Still on the role step: a pick being changed, or typed, is not taken.
        ({"currentStep": "role", "completedSteps": ["team", "autopilot"]}, None),
        ({"currentStep": "role"}, None),
        # Not answered yet.
        ({"currentStep": "autopilot", "completedSteps": ["team"]}, None),
        ({"role": ""}, None),
        ({"role": "Other", "otherRole": "   "}, None),
    ],
    ids=[
        "unchanged-option",
        "unchanged-other",
        "on-role-step",
        "back-on-role-step",
        "before-role-step",
        "empty",
        "blank-other",
    ],
)
async def test_nothing_is_sent_without_a_new_answered_pick(
    mocker, role_writes, fields, kept
):
    response = await _save_draft(mocker, _draft(**fields), kept=kept)

    assert response.status_code == 200
    role_writes["save"].assert_not_awaited()
    role_writes["queue"].assert_not_awaited()
    role_writes["posthog"].assert_not_called()


@pytest.mark.asyncio
async def test_the_draft_is_saved_when_keeping_the_pick_fails(
    mocker, role_writes, caplog
):
    role_writes["save"].side_effect = RuntimeError("database down")

    response = await _save_draft(mocker, _draft())

    assert response.status_code == 200
    assert response.json()["userId"] == USER_ID
    assert f"Could not keep the onboarding role of {USER_ID}" in caplog.text
    role_writes["posthog"].assert_not_called()


@pytest.mark.asyncio
async def test_a_conflicting_draft_sends_nothing(mocker, role_writes):
    mocker.patch.object(
        routes,
        "update_user_onboarding",
        AsyncMock(side_effect=OnboardingWizardConflict()),
    )

    response = await _patch(_draft())

    assert response.status_code == 409
    role_writes["save"].assert_not_awaited()
    role_writes["posthog"].assert_not_called()
