"""Route tests for POST /sessions/{session_id}/continue (the Paused card's Keep going)."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.chat import continuation as continuation_routes
from backend.api.features.chat import routes as chat_routes
from backend.copilot import model as chat_model
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.at_reset import cancel_resume_at_reset_in_background
from backend.copilot.constants import COPILOT_SYSTEM_PREFIX
from backend.copilot.continuation import KEEP_GOING_NOTE
from backend.copilot.executor import utils as executor_utils
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.rate_limit import RateLimitExceeded, RateLimitUnavailable
from backend.copilot.trial_cost_context import TrialCostContext
from backend.copilot.turn_pause import PAUSE_KEY, TurnPause, append_pause_marker

app = fastapi.FastAPI()
app.include_router(continuation_routes.router)
client = fastapi.testclient.TestClient(app)

# The real dispatch, kept before any test patches it.
DISPATCH_TURN = executor_utils.dispatch_turn

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
    mocks.rows = mocker.patch.object(
        continuation_routes, "get_chat_session", new=AsyncMock(return_value=None)
    )
    return mocks


def test_keep_going_starts_a_continuation_turn(chat):
    response = client.post(URL, json={"model": "standard"})

    assert response.status_code == 202
    assert response.json() == {"turn_id": "t-1"}
    kwargs = chat.schedule.await_args.kwargs
    assert kwargs["message"] == KEEP_GOING_NOTE
    assert kwargs["is_user_message"] is False
    assert kwargs["require_fresh_slot"] is True
    assert kwargs["dedupe_content"] is False
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


def test_a_microsoft_365_chat_is_refused_and_nothing_is_sent(chat):
    """Keep going would hand Microsoft our system note as the user's message,
    on a connection with no tools and none of the task."""
    chat.session.return_value = _session(route="microsoft_365_copilot")

    response = client.post(URL)

    assert response.status_code == 409
    assert response.json()["detail"] == "unsupported_connection"
    chat.schedule.assert_not_awaited()
    chat.publish.assert_not_awaited()


def test_a_full_concurrency_cap_is_a_429(chat):
    chat.schedule.side_effect = ConcurrentTurnLimitError("5 turns running")

    assert client.post(URL).status_code == 429


def test_a_chat_taken_after_the_busy_check_is_a_409(chat):
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


_PAUSE = f"{COPILOT_SYSTEM_PREFIX} Paused: checking in after a long run."


@pytest.fixture
def stored(mocker: pytest_mock.MockerFixture, chat: MagicMock) -> MagicMock:
    """The real scheduler and row store behind the route, over rows in memory.

    Mocked at the boundaries: the turn slot, the session lock, the queue and
    the database.
    """
    mocker.patch.object(
        continuation_routes,
        "schedule_chat_turn",
        new=executor_utils.schedule_chat_turn,
    )
    io = MagicMock()
    io.slot = MagicMock(admitted=True)
    io.session = ChatSession.new("test-user-id", dry_run=False)
    io.session.messages = [
        ChatMessage(role="user", content="Build the report"),
        ChatMessage(role="assistant", content=_PAUSE),
    ]

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield io.slot

    @asynccontextmanager
    async def session_lock(_session_id: str):
        yield True

    mocker.patch.object(executor_utils, "acquire_turn_slot", new=acquire)
    io.dispatch = mocker.patch.object(executor_utils, "dispatch_turn", new=AsyncMock())
    mocker.patch.object(chat_model, "_get_session_lock", new=session_lock)
    mocker.patch.object(
        chat_model, "get_chat_session", new=AsyncMock(return_value=io.session)
    )
    chat.rows.return_value = io.session
    io.save = mocker.patch.object(chat_model, "_save_session_to_db", new=AsyncMock())
    mocker.patch.object(
        chat_model,
        "chat_db",
        return_value=MagicMock(get_next_sequence=AsyncMock(return_value=2)),
    )
    mocker.patch.object(chat_model, "cache_chat_session", new=AsyncMock())
    return io


def _notes(session: ChatSession) -> int:
    return [m.content for m in session.messages].count(KEEP_GOING_NOTE)


def test_a_chat_another_turn_holds_is_a_409_and_saves_nothing(stored):
    # /stream and /continue both found the chat idle; /stream reserved it
    # first, so this request only refreshed the slot.
    stored.slot.admitted = False

    assert client.post(URL).status_code == 409
    assert _notes(stored.session) == 0
    stored.save.assert_not_awaited()
    stored.dispatch.assert_not_awaited()


def test_a_double_click_starts_one_turn(stored):
    first = client.post(URL)
    stored.slot.admitted = False  # the first click's turn holds the chat now
    second = client.post(URL)

    assert (first.status_code, second.status_code) == (202, 409)
    assert _notes(stored.session) == 1
    stored.dispatch.assert_awaited_once()


@pytest.mark.parametrize(
    "rows_after_the_note",
    [
        pytest.param(["Two of three sections done.", _PAUSE], id="assistant-rows"),
        pytest.param([], id="nothing"),
    ],
)
def test_keep_going_again_after_a_continuation_left_no_user_row(
    stored, rows_after_the_note: list[str]
):
    # The earlier note is still in the chat's trailing assistant run.
    stored.session.messages += [
        ChatMessage(role="assistant", content=KEEP_GOING_NOTE),
        *(ChatMessage(role="assistant", content=row) for row in rows_after_the_note),
    ]

    response = client.post(URL)

    assert response.status_code == 202
    assert stored.session.messages[-1].content == KEEP_GOING_NOTE
    assert _notes(stored.session) == 2
    stored.dispatch.assert_awaited_once()


def test_keep_going_is_not_marked_impersonated_by_default(chat):
    assert client.post(URL).status_code == 202
    assert chat.schedule.await_args.kwargs["impersonated"] is False


def test_an_impersonating_admins_keep_going_is_marked(
    chat, mock_jwt_admin, target_user_id: str
):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]

    response = client.post(URL, headers={"X-Act-As-User-Id": target_user_id})

    assert response.status_code == 202
    kwargs = chat.schedule.await_args.kwargs
    assert kwargs["user_id"] == target_user_id
    assert kwargs["impersonated"] is True


@pytest.fixture
def queued(mocker: pytest_mock.MockerFixture, stored: MagicMock) -> MagicMock:
    """``stored`` with the real dispatch too, down to the queue: what the
    executor would pick up is ``queued.entry()``."""
    mocker.patch.object(executor_utils, "dispatch_turn", new=DISPATCH_TURN)
    mocker.patch.object(executor_utils, "admit_turn", new=AsyncMock())
    mocker.patch.object(
        executor_utils,
        "capture_cost_context",
        new=AsyncMock(return_value=TrialCostContext(user_id="u1", trial_id=None)),
    )
    mocker.patch("backend.copilot.stream_registry.create_session", new=AsyncMock())
    queue = MagicMock(publish_message=AsyncMock())
    mocker.patch(
        "backend.util.clients.get_async_copilot_queue",
        new=AsyncMock(return_value=queue),
    )

    def entry() -> executor_utils.CoPilotExecutionEntry:
        call = queue.publish_message.await_args
        assert call is not None
        return executor_utils.CoPilotExecutionEntry.model_validate_json(
            call.kwargs["message"]
        )

    stored.entry = entry
    return stored


def _paused_rows(*, tainted: bool) -> list[ChatMessage]:
    session = ChatSession.new("test-user-id", dry_run=False)
    session.messages = [ChatMessage(role="user", content="Build the report")]
    append_pause_marker(session, TurnPause(reason="steps", steps=450), tainted=tainted)
    return session.messages


@pytest.mark.parametrize(
    "tainted",
    [True, False],
    ids=["paused-turn-from-a-bot", "paused-turn-from-the-owner"],
)
def test_keep_going_starts_with_the_paused_turns_taint(queued, tainted: bool):
    """A web chat, so only the paused turn's own record can say a bot's
    message started the work Keep going carries on."""
    queued.session.messages = _paused_rows(tainted=tainted)

    assert client.post(URL).status_code == 202

    entry = queued.entry()
    assert entry.message == KEEP_GOING_NOTE
    assert entry.envelope is not None
    assert entry.envelope.tainted is tainted


def test_keep_going_on_a_marker_from_before_the_field_continues_tainted(queued):
    """An executor from before the field wrote it, so how the paused turn ran
    is unknown: Keep going runs as if it was tainted."""
    queued.session.messages = [
        ChatMessage(role="user", content="Build the report"),
        ChatMessage(
            role="assistant",
            content=_PAUSE,
            metadata={PAUSE_KEY: TurnPause(reason="steps", steps=450).as_part()},
        ),
    ]

    assert client.post(URL).status_code == 202

    entry = queued.entry()
    assert entry.envelope is not None
    assert entry.envelope.tainted is True
