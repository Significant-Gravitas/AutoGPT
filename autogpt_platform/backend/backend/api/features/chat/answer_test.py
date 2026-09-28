"""POST /sessions/{id}/messages: answer a thread without opening its stream."""

from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.chat import answer
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.pending_message_helpers import QueuePendingMessageResponse
from backend.util.exceptions import NotFoundError

app = fastapi.FastAPI()
app.include_router(answer.router)


@app.exception_handler(NotFoundError)
async def _not_found(request: fastapi.Request, exc: NotFoundError):
    return fastapi.responses.JSONResponse(status_code=404, content={"detail": str(exc)})


client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def seams(mocker: pytest_mock.MockerFixture):
    session = MagicMock(session_id="sub-1", expert_id=None, organization_id=None)
    session.metadata.llm_auth_provider = "platform"
    session.metadata.llm_credential_id = None
    session.metadata.origin = "interactive"
    mocks = {
        "session": session,
        "meta": mocker.patch.object(
            answer, "get_chat_session_metadata", AsyncMock(return_value=session)
        ),
        "in_flight": mocker.patch.object(
            answer, "is_turn_in_flight", AsyncMock(return_value=False)
        ),
        "schedule": mocker.patch.object(
            answer, "schedule_chat_turn", AsyncMock(return_value="turn-1")
        ),
        "queue_pending": mocker.patch.object(
            answer,
            "queue_pending_for_http",
            AsyncMock(
                return_value=QueuePendingMessageResponse(
                    buffer_length=1, max_buffer_length=10, turn_in_flight=True
                )
            ),
        ),
        "enqueue_turn": mocker.patch.object(answer.turn_queue, "try_enqueue_turn"),
        "clear": mocker.patch.object(
            answer, "clear_session_pending_question", AsyncMock()
        ),
        "invalidate": mocker.patch.object(
            answer, "invalidate_session_cache", AsyncMock()
        ),
    }
    mocker.patch.object(answer, "enforce_payment_paywall", AsyncMock())
    mocker.patch.object(
        answer, "get_global_rate_limits", AsyncMock(return_value=(0, 0, None))
    )
    mocker.patch.object(answer, "check_rate_limit", AsyncMock())
    mocker.patch.object(answer, "resolve_session_permissions", return_value=None)
    return mocks


def test_an_idle_thread_starts_a_turn_with_the_answer(seams, test_user_id):
    response = client.post("/sessions/sub-1/messages", json={"message": "Q4"})

    assert response.status_code == 200
    assert response.json() == {"session_id": "sub-1", "queued": False}
    kwargs = seams["schedule"].await_args.kwargs
    assert (kwargs["session_id"], kwargs["user_id"]) == ("sub-1", test_user_id)
    assert kwargs["message"] == "Q4"
    assert kwargs["is_user_message"] is True
    seams["meta"].assert_awaited_once_with("sub-1", test_user_id)
    seams["clear"].assert_awaited_once_with("sub-1", test_user_id)


def test_a_busy_thread_takes_the_answer_into_its_running_turn(seams):
    seams["in_flight"].return_value = True

    response = client.post("/sessions/sub-1/messages", json={"message": "Q4"})

    assert response.status_code == 200
    assert response.json() == {"session_id": "sub-1", "queued": True}
    seams["queue_pending"].assert_awaited_once()
    seams["schedule"].assert_not_awaited()


def test_at_the_running_cap_the_turn_waits_in_the_queue(seams):
    seams["schedule"].side_effect = ConcurrentTurnLimitError("cap")

    response = client.post("/sessions/sub-1/messages", json={"message": "Q4"})

    assert response.status_code == 200
    assert response.json()["queued"] is True
    seams["enqueue_turn"].assert_awaited_once()


def test_someone_elses_thread_is_not_found(seams):
    seams["meta"].return_value = None

    response = client.post("/sessions/sub-1/messages", json={"message": "Q4"})

    assert response.status_code == 404
    seams["schedule"].assert_not_awaited()


def test_an_empty_answer_is_rejected(seams):
    response = client.post("/sessions/sub-1/messages", json={"message": "  "})

    assert response.status_code == 422
