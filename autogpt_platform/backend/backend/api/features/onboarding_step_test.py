from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features.v1 import v1_router
from backend.data.onboarding import OnboardingStep

app = fastapi.FastAPI()
app.include_router(v1_router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.mark.parametrize("invalid_step", ["VISIT_COPILOT", "BOGUS", "welcome", ""])
def test_complete_step_rejects_invalid_step(invalid_step):
    # Boundary validation is what replaces the dropped Prisma enum: any value
    # outside FrontendOnboardingStep must be rejected before any DB write. This
    # also locks in that the retired VISIT_COPILOT value is no longer accepted.
    response = client.post("/onboarding/step", params={"step": invalid_step})
    assert response.status_code == 422


def test_complete_step_accepts_renamed_complete_value(mocker):
    mock_complete = mocker.patch(
        "backend.api.features.v1.complete_onboarding_step",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.post(
        "/onboarding/step",
        params={"step": OnboardingStep.ONBOARDING_COMPLETE.value},
    )

    assert response.status_code == 200
    mock_complete.assert_awaited_once()
    assert mock_complete.await_args.args[1] == OnboardingStep.ONBOARDING_COMPLETE


@pytest.mark.parametrize(
    "step",
    [
        OnboardingStep.AGENTS_TAB_INTRO,
        OnboardingStep.MARKETPLACE_TAB_INTRO,
        OnboardingStep.BUILD_TAB_INTRO,
    ],
)
def test_complete_step_accepts_tab_intros(step, mocker):
    # Each tab's first-visit card records its own step; without all three on
    # FrontendOnboardingStep the card would 422 and reappear forever.
    mock_complete = mocker.patch(
        "backend.api.features.v1.complete_onboarding_step",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.post("/onboarding/step", params={"step": step.value})

    assert response.status_code == 200
    assert mock_complete.await_args.args[1] == step


@pytest.mark.parametrize(
    "step",
    [
        OnboardingStep.MARKETPLACE_ADD_AGENT,
        OnboardingStep.LIBRARY_RUN_AGENT,
        OnboardingStep.RUN_AGENTS_100,
    ],
)
def test_complete_step_rejects_rewarded_backend_only_steps(step, mocker):
    # These are real OnboardingStep values that carry a credit reward and are
    # deliberately left off FrontendOnboardingStep. The endpoint's Literal is
    # the only thing standing between an authenticated user and self-awarding
    # credits, so it gets its own test.
    mock_complete = mocker.patch(
        "backend.api.features.v1.complete_onboarding_step",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.post("/onboarding/step", params={"step": step.value})

    assert response.status_code == 422
    mock_complete.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step",
    [
        OnboardingStep.AGENTS_TAB_INTRO,
        OnboardingStep.MARKETPLACE_TAB_INTRO,
        OnboardingStep.BUILD_TAB_INTRO,
    ],
)
async def test_tab_intro_steps_grant_no_reward(step, mocker):
    # The tab intros are client-writable, so "unrewarded" has to be enforced
    # rather than documented: posting one must never reach the credit model.
    from backend.data import onboarding as onboarding_module
    from backend.data.model import UserOnboarding

    mock_credit_model = mocker.patch.object(
        onboarding_module,
        "get_user_credit_model",
        new_callable=AsyncMock,
    )

    await onboarding_module._reward_user(
        "test-user-id",
        UserOnboarding.model_construct(rewardedFor=[]),
        step,
    )

    mock_credit_model.assert_not_awaited()


def test_is_onboarding_completed_true_when_complete_step_present(mocker):
    from backend.data.model import UserOnboarding

    mock_get = mocker.patch(
        "backend.api.features.v1.get_user_onboarding",
        new_callable=AsyncMock,
    )
    mock_get.return_value = UserOnboarding.model_construct(
        completedSteps=[OnboardingStep.WELCOME, OnboardingStep.ONBOARDING_COMPLETE],
    )

    response = client.get("/onboarding/completed")

    assert response.status_code == 200
    assert response.json()["is_completed"] is True


def test_is_onboarding_completed_false_without_complete_step(mocker):
    from backend.data.model import UserOnboarding

    mock_get = mocker.patch(
        "backend.api.features.v1.get_user_onboarding",
        new_callable=AsyncMock,
    )
    mock_get.return_value = UserOnboarding.model_construct(
        completedSteps=[OnboardingStep.WELCOME],
    )

    response = client.get("/onboarding/completed")

    assert response.status_code == 200
    assert response.json()["is_completed"] is False


def test_user_onboarding_response_model_accepts_deprecated_stored_values():
    from backend.data.model import UserOnboarding

    # Deprecated steps stay in OnboardingStep forever: existing rows still
    # contain them and the strict ``list[OnboardingStep]`` response model would
    # 500 reads otherwise. This validates (no ``model_construct``) so it fails
    # if a member is ever deleted from the enum.
    onboarding = UserOnboarding.model_validate(
        {
            "userId": "test-user-id",
            "completedSteps": ["MARKETPLACE_VISIT", "RE_RUN_AGENT", "RUN_AGENTS"],
            "walletShown": False,
            "notified": ["BUILDER_SAVE_AGENT"],
            "rewardedFor": ["MARKETPLACE_VISIT"],
            "usageReason": None,
            "integrations": [],
            "otherIntegrations": None,
            "selectedStoreListingVersionId": None,
            "agentInput": None,
            "onboardingAgentExecutionId": None,
            "agentRuns": 0,
            "lastRunAt": None,
            "consecutiveRunDays": 0,
        }
    )

    assert onboarding.completedSteps == [
        OnboardingStep.MARKETPLACE_VISIT,
        OnboardingStep.RE_RUN_AGENT,
        OnboardingStep.RUN_AGENTS,
    ]


def test_update_onboarding_rejects_invalid_notified_step(mocker):
    mock_update = mocker.patch(
        "backend.api.features.v1.update_user_onboarding",
        new_callable=AsyncMock,
    )

    response = client.patch("/onboarding", json={"notified": ["VISIT_COPILOT"]})

    assert response.status_code == 422
    mock_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_user_onboarding_merges_notified_as_plain_strings(mocker):
    from backend.data import onboarding as onboarding_module
    from backend.data.model import UserOnboarding

    mocker.patch.object(
        onboarding_module,
        "get_user_onboarding",
        new_callable=AsyncMock,
        return_value=UserOnboarding.model_construct(notified=["WELCOME"]),
    )
    mock_prisma_model = mocker.patch.object(onboarding_module, "UserOnboarding")
    mock_upsert = AsyncMock()
    mock_prisma_model.prisma.return_value.upsert = mock_upsert

    await onboarding_module.update_user_onboarding(
        "test-user-id",
        onboarding_module.UserOnboardingUpdate(
            notified=[OnboardingStep.AGENT_INPUT, OnboardingStep.WELCOME]
        ),
    )

    notified = mock_upsert.await_args.kwargs["data"]["update"]["notified"]
    assert sorted(notified) == ["AGENT_INPUT", "WELCOME"]
    assert all(type(value) is str for value in notified)
