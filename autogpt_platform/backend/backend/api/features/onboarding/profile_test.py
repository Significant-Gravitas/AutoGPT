from unittest.mock import AsyncMock, MagicMock

import fastapi
import httpx
import pytest
from autogpt_libs.auth import get_user_id, requires_user

from backend.api.features.onboarding import routes
from backend.api.features.onboarding.routes import router
from backend.data.onboarding_role import OnboardingRole
from backend.data.understanding import BusinessUnderstanding, BusinessUnderstandingInput

app = fastapi.FastAPI()
app.include_router(router)

USER_ID = "user-1"


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as session:
        yield session


@pytest.fixture(autouse=True)
def profile_dependencies(mocker):
    existing = mocker.patch.object(
        routes, "get_business_understanding", new=AsyncMock(return_value=None)
    )
    admission = mocker.patch.object(
        routes, "enforce_personalization_budget", new=AsyncMock()
    )
    return existing, admission


@pytest.fixture(autouse=True)
def role_writes(mocker):
    """The kept pick's copy, and its MailerLite and PostHog writes."""
    return {
        "save": mocker.patch.object(routes, "save_onboarding_role", new=AsyncMock()),
        "queue": mocker.patch.object(routes, "queue_onboarding_role", new=AsyncMock()),
        "posthog": mocker.patch.object(routes, "set_onboarding_role", new=MagicMock()),
    }


@pytest.fixture(autouse=True)
def setup_app_auth():
    async def authenticated_user():
        return USER_ID

    async def authenticated():
        return None

    app.dependency_overrides[get_user_id] = authenticated_user
    app.dependency_overrides[requires_user] = authenticated
    yield
    app.dependency_overrides.clear()


async def test_onboarding_profile_success(mocker, client):
    mock_extract = mocker.patch(
        "backend.api.features.onboarding.routes.extract_business_understanding",
        new_callable=AsyncMock,
    )
    mock_upsert = mocker.patch(
        "backend.api.features.onboarding.routes.upsert_business_understanding",
        new_callable=AsyncMock,
    )

    mock_extract.return_value = BusinessUnderstandingInput.model_construct(
        user_name="John",
        user_role="Founder/CEO",
        pain_points=["Finding leads"],
        suggested_prompts={"Learn": ["How do I automate lead gen?"]},
    )
    mock_upsert.return_value = AsyncMock()

    response = await client.post(
        "/onboarding/profile",
        json={
            "user_name": "John",
            "user_role": "Founder/CEO",
            "pain_points": ["Finding leads", "Email & outreach"],
        },
    )
    assert response.status_code == 200
    mock_extract.assert_awaited_once()
    mock_upsert.assert_awaited_once()


@pytest.mark.parametrize(
    "user_role, kept",
    [
        ("Founder/CEO", OnboardingRole(choice="Founder/CEO")),
        ("Dentist", OnboardingRole(choice="Other", other="Dentist")),
    ],
)
async def test_the_pick_is_kept_and_sent_to_mailerlite_and_posthog(
    mocker, role_writes, user_role, kept, client
):
    # The extraction's own idea of the role, which AutoPilot-side rewrites
    # look like; the kept pick comes from the request.
    mocker.patch.object(
        routes,
        "extract_business_understanding",
        new=AsyncMock(
            return_value=BusinessUnderstandingInput.model_construct(
                user_role="decision maker"
            )
        ),
    )
    mocker.patch.object(routes, "upsert_business_understanding", new=AsyncMock())
    response = await client.post(
        "/onboarding/profile",
        json={
            "user_name": "John",
            "user_role": user_role,
            "pain_points": ["Finding leads"],
        },
    )
    assert response.status_code == 200
    role_writes["save"].assert_awaited_once_with(USER_ID, kept)
    role_writes["queue"].assert_awaited_once_with(USER_ID, kept)
    role_writes["posthog"].assert_called_once_with(user_id=USER_ID, role=kept)


async def test_an_unchanged_profile_still_keeps_the_pick(
    profile_dependencies, role_writes, client
):
    """A retry after a lost write, or a profile first saved before the pick
    was kept, still keeps it."""
    existing, _ = profile_dependencies
    existing.return_value = BusinessUnderstanding.model_construct(
        user_name="John", user_role="Dentist", pain_points=["Finding leads"]
    )
    response = await client.post(
        "/onboarding/profile",
        json={
            "user_name": "John",
            "user_role": "Dentist",
            "pain_points": ["Finding leads"],
        },
    )
    assert response.status_code == 200
    kept = OnboardingRole(choice="Other", other="Dentist")
    role_writes["save"].assert_awaited_once_with(USER_ID, kept)
    role_writes["queue"].assert_awaited_once_with(USER_ID, kept)
    role_writes["posthog"].assert_called_once_with(user_id=USER_ID, role=kept)


async def test_onboarding_profile_missing_fields(client):
    response = await client.post(
        "/onboarding/profile",
        json={"user_name": "John"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("length, status", [(2000, 200), (2001, 422)])
async def test_profile_pain_point_size_is_bounded(
    mocker, profile_dependencies, length, status, client
):
    _, admission = profile_dependencies
    extract = mocker.patch.object(
        routes,
        "extract_business_understanding",
        new=AsyncMock(return_value=BusinessUnderstandingInput.model_construct()),
    )
    mocker.patch.object(routes, "upsert_business_understanding", new=AsyncMock())
    response = await client.post(
        "/onboarding/profile",
        json={
            "user_name": "John",
            "user_role": "Founder",
            "pain_points": ["a" * length],
        },
    )
    assert response.status_code == status
    if status == 422:
        admission.assert_not_awaited()
        extract.assert_not_awaited()


async def test_identical_saved_profile_does_not_repeat_personalization(
    mocker, profile_dependencies, client
):
    existing, admission = profile_dependencies
    existing.return_value = BusinessUnderstanding.model_construct(
        user_name="John", user_role="Founder/CEO", pain_points=["Finding leads"]
    )
    extract = mocker.patch.object(
        routes, "extract_business_understanding", new=AsyncMock()
    )
    response = await client.post(
        "/onboarding/profile",
        json={
            "user_name": "John",
            "user_role": "Founder/CEO",
            "pain_points": ["Finding leads"],
        },
    )
    assert response.status_code == 200
    admission.assert_not_awaited()
    extract.assert_not_awaited()


@pytest.mark.parametrize(
    "changed",
    [{"user_name": "Jane"}, {"user_role": "Developer"}, {"pain_points": ["Email"]}],
)
async def test_changed_profile_still_checks_budget(
    mocker, profile_dependencies, changed, client
):
    existing, admission = profile_dependencies
    original = {
        "user_name": "John",
        "user_role": "Founder/CEO",
        "pain_points": ["Finding leads"],
    }
    existing.return_value = BusinessUnderstanding.model_construct(**original)
    admission.side_effect = fastapi.HTTPException(429, "Try again")
    extract = mocker.patch.object(
        routes, "extract_business_understanding", new=AsyncMock()
    )
    response = await client.post("/onboarding/profile", json={**original, **changed})
    assert response.status_code == 429
    admission.assert_awaited_once()
    extract.assert_not_awaited()


@pytest.mark.parametrize("status", [429, 503])
async def test_profile_rejection_is_not_swallowed_by_extraction_fallback(
    mocker, profile_dependencies, status, client
):
    _, admission = profile_dependencies
    admission.side_effect = fastapi.HTTPException(
        status, "Try again", headers={"Retry-After": "30"}
    )
    extract = mocker.patch.object(
        routes, "extract_business_understanding", new=AsyncMock()
    )
    upsert = mocker.patch.object(
        routes, "upsert_business_understanding", new=AsyncMock()
    )
    response = await client.post(
        "/onboarding/profile", json={"user_name": "John", "user_role": "Founder/CEO"}
    )
    assert response.status_code == status
    assert response.headers["Retry-After"] == "30"
    extract.assert_not_awaited()
    upsert.assert_not_awaited()
