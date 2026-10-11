"""Route tests for POST /sessions/{session_id}/continue (the Paused card's Keep going)."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.chat import continuation as continuation_routes
from backend.api.features.chat import routes as chat_routes
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.at_reset import cancel_resume_at_reset_in_background
from backend.copilot.continuation import KEEP_GOING_NOTE
from backend.copilot.rate_limit import RateLimitExceeded, RateLimitUnavailable

app = fastapi.FastAPI()
app.include_router(continuation_routes.router)
client = fastapi.testclient.TestClient(app)

SESSION_ID = "8d2f5c1e-2b7a-4c55-9d0e-3f6a1b2c4d5e"
URL = f"/sessions/{SESSION_ID}/continue"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


def _session(route: str = "platform") -> MagicMock:
    session = MagicMock()
    session.expert_id = None
    session.organization_id = None
    session.team_id = None
    session.metadata.llm_auth_provider = route
    session.metadata.llm_credential_id = None
    session.metadata.origin = "interactive"
    session.metadata.source_platform = None
    return session


@pytest.fixture
def chat(mocker: pytest_mock.MockerFixture):
    """Patch everything the route touches; tests flip what they need."""
    mocks = MagicMock()
    mocks.session = mocker.patch.object(
        continuation_routes,
        "_validate_and_get_writable_session",
        new=AsyncMock(return_value=_session()),
    )
    mocks.paywall = mocker.patch.object(
        continuation_routes, "enforce_payment_paywall", new=AsyncMock()
    )
    mocks.codex_access = mocker.patch.object(
        continuation_routes, "enforce_codex_access_http", new=AsyncMock()
    )
    mocker.patch.object(
        chat_routes,
        "get_global_rate_limits",
        new=AsyncMock(return_value=(1_000_000, 5_000_000, "PRO")),
    )
    mocks.check_rate_limit = mocker.patch.object(
        chat_routes, "check_rate_limit", new=AsyncMock()
    )
    mocks.busy = mocker.patch.object(
        continuation_routes, "is_turn_in_flight", new=AsyncMock(return_value=False)
    )
    mocks.schedule = mocker.patch.object(
        continuation_routes, "schedule_chat_turn", new=AsyncMock(return_value="t-1")
    )
    mocker.patch.object(
        continuation_routes, "resolve_session_permissions", return_value=None
    )
    mocks.publish = mocker.patch.object(
        continuation_routes.stream_registry, "publish_chunk", new=AsyncMock()
    )
    mocks.cancel_resume = mocker.patch.object(
        continuation_routes, "cancel_resume_at_reset_in_background"
    )
    return mocks


def test_keep_going_starts_a_continuation_turn(chat):
    response = client.post(URL, json={"model": "standard"})

    assert response.status_code == 202
    assert response.json() == {"turn_id": "t-1"}
    kwargs = chat.schedule.await_args.kwargs
    assert kwargs["message"] == KEEP_GOING_NOTE
    assert kwargs["is_user_message"] is False
    assert kwargs["model"] == "standard"
    chat.paywall.assert_awaited_once()
    chat.check_rate_limit.assert_awaited_once()
    chat.publish.assert_awaited_once()


def test_body_is_optional(chat):
    assert client.post(URL).status_code == 202
    assert chat.schedule.await_args.kwargs["model"] is None


def test_a_running_turn_is_not_doubled(chat):
    chat.busy.return_value = True

    response = client.post(URL)

    assert response.status_code == 409
    chat.schedule.assert_not_awaited()


def test_an_out_of_budget_user_gets_the_usage_limit_envelope(chat):
    resets = datetime.now(UTC) + timedelta(hours=4)
    chat.check_rate_limit.side_effect = RateLimitExceeded("daily", resets)

    response = client.post(URL)

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["kind"] == "usage_limit"
    assert detail["resetsAt"] == int(resets.timestamp())
    chat.schedule.assert_not_awaited()


def test_an_unreadable_allowance_fails_closed(chat):
    chat.check_rate_limit.side_effect = RateLimitUnavailable()

    response = client.post(URL)

    assert response.status_code == 503
    chat.schedule.assert_not_awaited()


def test_a_chatgpt_chat_skips_the_platform_allowance(chat):
    chat.session.return_value = _session(route="codex")

    assert client.post(URL).status_code == 202
    chat.codex_access.assert_awaited_once()
    chat.paywall.assert_not_awaited()
    chat.check_rate_limit.assert_not_awaited()


def test_a_full_concurrency_cap_is_a_429(chat):
    chat.schedule.side_effect = ConcurrentTurnLimitError("5 turns running")

    assert client.post(URL).status_code == 429


def test_a_duplicate_click_is_a_409(chat):
    chat.schedule.return_value = None

    assert client.post(URL).status_code == 409
    chat.publish.assert_not_awaited()


def test_keep_going_cancels_a_pending_resume_at_reset(chat, test_user_id):
    # Only the resume: ``cancel_resume_at_reset_in_background`` leaves a
    # pending send alone (copilot/at_reset_test.py).
    assert client.post(URL).status_code == 202
    chat.cancel_resume.assert_called_once_with(SESSION_ID, test_user_id)


@pytest.mark.parametrize("refusal", ["busy", "duplicate"])
def test_a_keep_going_that_didnt_start_leaves_the_resume(chat, refusal):
    if refusal == "busy":
        chat.busy.return_value = True
    else:
        chat.schedule.return_value = None

    assert client.post(URL).status_code == 409
    chat.cancel_resume.assert_not_called()


def test_a_failing_resume_cancel_never_fails_keep_going(
    chat, mocker: pytest_mock.MockerFixture
):
    mocker.patch.object(
        continuation_routes,
        "cancel_resume_at_reset_in_background",
        cancel_resume_at_reset_in_background,
    )
    scheduler_down = MagicMock(
        cancel_at_reset_turn=AsyncMock(side_effect=RuntimeError("scheduler down"))
    )
    mocker.patch(
        "backend.copilot.at_reset.get_scheduler_client", return_value=scheduler_down
    )

    assert client.post(URL).status_code == 202
    chat.schedule.assert_awaited_once()
