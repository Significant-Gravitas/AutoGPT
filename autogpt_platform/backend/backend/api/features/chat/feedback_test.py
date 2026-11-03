"""Route tests for POST /sessions/{session_id}/feedback (thumbs up/down, copy).

The database layer and PostHog are mocked at the boundary;
``backend/copilot/feedback_db_test.py`` proves the owner scoping against a
real Postgres.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.models import ChatMessage as PrismaChatMessage

from backend.api.features.chat import feedback as feedback_routes
from backend.copilot import feedback as feedback_service
from backend.copilot import feedback_db

app = fastapi.FastAPI()
app.include_router(feedback_routes.router)
client = fastapi.testclient.TestClient(app)

SESSION_ID = "8d2f5c1e-2b7a-4c55-9d0e-3f6a1b2c4d5e"
MESSAGE_ROW_ID = "0f9e8d7c-6b5a-4938-8271-605f4e3d2c1b"
FEEDBACK_ID = "c3a1b2d4-e5f6-4789-8abc-def012345678"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def posthog_capture(mocker: pytest_mock.MockerFixture) -> MagicMock:
    """PostHog is covered in ``backend/copilot/feedback_test.py``; here it
    only must not reach a real client."""
    return mocker.patch.object(feedback_service, "capture")


@pytest.fixture
def upsert(mocker: pytest_mock.MockerFixture) -> AsyncMock:
    return mocker.patch.object(
        feedback_db,
        "upsert_message_feedback",
        new=AsyncMock(return_value=FEEDBACK_ID),
    )


def _reply(*, sequence: int = 7) -> PrismaChatMessage:
    return PrismaChatMessage(
        id=MESSAGE_ROW_ID,
        createdAt=datetime.now(UTC),
        sessionId=SESSION_ID,
        role="assistant",
        content="Here is the report.",
        sequence=sequence,
    )


def _mock_message(
    mocker: pytest_mock.MockerFixture, message: PrismaChatMessage | None
) -> AsyncMock:
    return mocker.patch.object(
        feedback_db, "get_rateable_message", new=AsyncMock(return_value=message)
    )


def _post(body: dict, session_id: str = SESSION_ID):
    return client.post(f"/sessions/{session_id}/feedback", json=body)


def _thumbs_down(comment: str | None = "It edited the wrong file") -> dict:
    return {
        "message_id": f"{SESSION_ID}-seq-7",
        "score_name": "user-feedback",
        "score_value": 0,
        "comment": comment,
    }


# ─── Access ────────────────────────────────────────────────────────────


def test_feedback_rejects_unauthenticated_request() -> None:
    unauthenticated_app = fastapi.FastAPI()
    unauthenticated_app.include_router(feedback_routes.router)
    unauthenticated_client = fastapi.testclient.TestClient(unauthenticated_app)

    response = unauthenticated_client.post(
        f"/sessions/{SESSION_ID}/feedback", json=_thumbs_down()
    )

    assert response.status_code == 401


def test_feedback_on_another_users_session_is_not_found(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    upsert: AsyncMock,
) -> None:
    """The lookup is scoped to the caller, so a session owned by someone else
    yields no message and the same 404 an unknown session would."""
    owner_id = "someone-else"

    async def owner_only(user_id: str, session_id: str, **_):
        return _reply() if user_id == owner_id else None

    lookup = mocker.patch.object(
        feedback_db, "get_rateable_message", new=AsyncMock(side_effect=owner_only)
    )

    response = _post(_thumbs_down())

    assert response.status_code == 404
    assert lookup.await_args is not None
    assert lookup.await_args.args == (test_user_id, SESSION_ID)
    upsert.assert_not_awaited()


def test_feedback_on_unknown_message_is_not_found(
    mocker: pytest_mock.MockerFixture,
    upsert: AsyncMock,
) -> None:
    _mock_message(mocker, None)

    response = _post(_thumbs_down())

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]
    upsert.assert_not_awaited()


# ─── Rating ────────────────────────────────────────────────────────────


def test_thumbs_down_is_saved(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    upsert: AsyncMock,
) -> None:
    lookup = _mock_message(mocker, _reply())

    response = _post(_thumbs_down())

    assert response.status_code == 200
    assert response.json() == {"id": FEEDBACK_ID}
    lookup.assert_awaited_once_with(test_user_id, SESSION_ID, sequence=7)
    upsert.assert_awaited_once_with(
        user_id=test_user_id,
        session_id=SESSION_ID,
        message_id=MESSAGE_ROW_ID,
        score_name="user-feedback",
        score_value=0,
        comment="It edited the wrong file",
    )


def test_copy_is_saved_under_its_own_name(
    mocker: pytest_mock.MockerFixture,
    upsert: AsyncMock,
) -> None:
    _mock_message(mocker, _reply())

    response = _post(
        {
            "message_id": f"{SESSION_ID}-seq-7",
            "score_name": "copy",
            "score_value": 1,
        }
    )

    assert response.status_code == 200
    assert upsert.await_args is not None
    assert upsert.await_args.kwargs["score_name"] == "copy"
    assert upsert.await_args.kwargs["score_value"] == 1


def test_raw_row_id_is_looked_up_by_id(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    upsert: AsyncMock,
) -> None:
    lookup = _mock_message(mocker, _reply())

    response = _post({**_thumbs_down(), "message_id": MESSAGE_ROW_ID})

    assert response.status_code == 200
    lookup.assert_awaited_once_with(test_user_id, SESSION_ID, message_id=MESSAGE_ROW_ID)


def test_database_failure_is_an_error_the_ui_can_show(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """With the rating unsaved the request fails rather than reporting a
    success the UI would thank the user for."""
    _mock_message(mocker, _reply())
    mocker.patch.object(
        feedback_db,
        "upsert_message_feedback",
        new=AsyncMock(side_effect=RuntimeError("db down")),
    )
    failing_client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)

    response = failing_client.post(
        f"/sessions/{SESSION_ID}/feedback", json=_thumbs_down()
    )

    assert response.status_code == 500


# ─── Validation ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"score_name": "stars"},
        {"score_value": 2},
        {"score_value": -1},
        {"score_value": 0.5},
        {"score_name": "copy", "score_value": 0},
        {"comment": "x" * 2001},
        {"message_id": ""},
    ],
    ids=[
        "unknown-score-name",
        "value-above-one",
        "negative-value",
        "fractional-value",
        "copy-scored-zero",
        "comment-too-long",
        "empty-message-id",
    ],
)
def test_invalid_ratings_are_rejected(
    mocker: pytest_mock.MockerFixture,
    overrides: dict,
    upsert: AsyncMock,
) -> None:
    lookup = _mock_message(mocker, _reply())

    response = _post({**_thumbs_down(), **overrides})

    assert response.status_code == 422
    lookup.assert_not_awaited()
    upsert.assert_not_awaited()


def test_rating_is_sent_to_posthog_as_the_user(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
    upsert: AsyncMock,
    posthog_capture: MagicMock,
) -> None:
    _mock_message(mocker, _reply())

    response = _post(_thumbs_down())

    assert response.status_code == 200
    distinct_id, event, properties = posthog_capture.call_args.args
    assert distinct_id == test_user_id
    assert event == "$ai_metric"
    assert properties["impersonated"] is False


def test_rating_by_an_impersonating_admin_is_marked(
    mocker: pytest_mock.MockerFixture,
    mock_jwt_admin,
    target_user_id: str,
    upsert: AsyncMock,
    posthog_capture: MagicMock,
) -> None:
    app.dependency_overrides[get_jwt_payload] = mock_jwt_admin["get_jwt_payload"]
    _mock_message(mocker, _reply())

    response = client.post(
        f"/sessions/{SESSION_ID}/feedback",
        json=_thumbs_down(),
        headers={"X-Act-As-User-Id": target_user_id},
    )

    assert response.status_code == 200
    distinct_id, _, properties = posthog_capture.call_args.args
    assert distinct_id == target_user_id
    assert properties["impersonated"] is True


def test_blank_comment_is_saved_as_none(
    mocker: pytest_mock.MockerFixture,
    upsert: AsyncMock,
) -> None:
    _mock_message(mocker, _reply())

    response = _post(_thumbs_down(comment="   "))

    assert response.status_code == 200
    assert upsert.await_args is not None
    assert upsert.await_args.kwargs["comment"] is None
