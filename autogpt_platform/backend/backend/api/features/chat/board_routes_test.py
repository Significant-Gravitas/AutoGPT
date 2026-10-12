"""Tests for the task board routes and board fields on POST /sessions."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pydantic
import pytest
import pytest_mock

from backend.api.features.chat import board_routes
from backend.api.features.chat import routes as chat_routes
from backend.api.features.chat import server_turn
from backend.copilot.board import DEFAULT_COLUMNS, validate_columns
from backend.copilot.board_db import BoardExtras, BoardSessionRow
from backend.copilot.model import ChatSession, ChatSessionInfo, ChatSessionMetadata
from backend.copilot.rate_limit import SubscriptionTier
from backend.util.exceptions import NotFoundError

app = fastapi.FastAPI()
app.include_router(board_routes.router)
app.include_router(chat_routes.router)


@app.exception_handler(NotFoundError)
async def _not_found_handler(
    request: fastapi.Request, exc: NotFoundError
) -> fastapi.responses.JSONResponse:
    """Mirror the production NotFoundError → 404 mapping from the REST app."""
    return fastapi.responses.JSONResponse(status_code=404, content={"detail": str(exc)})


client = fastapi.testclient.TestClient(app)

TEST_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
SESSION_ID = "sess-1"


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, mocker: pytest_mock.MockerFixture):
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    mocker.patch.object(
        board_routes, "processing_session_ids", new=AsyncMock(return_value=set())
    )
    yield
    app.dependency_overrides.clear()


def _session(
    session_id: str = SESSION_ID,
    *,
    title: str | None = "Write the PRD",
    expert_id: str | None = None,
    metadata: ChatSessionMetadata | None = None,
) -> ChatSessionInfo:
    return ChatSessionInfo(
        session_id=session_id,
        user_id=TEST_USER_ID,
        title=title,
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=metadata or ChatSessionMetadata(),
        expert_id=expert_id,
    )


def _row(session: ChatSessionInfo, **overrides) -> BoardSessionRow:
    fields = {
        "board_column": "in_progress",
        "board_position": 0.0,
        "started": True,
        "needs_you": False,
        "last_line": "Drafted the outline.",
        "file_count": 2,
    }
    return BoardSessionRow(session=session, **{**fields, **overrides})


def _default_columns():
    return [c.model_copy() for c in DEFAULT_COLUMNS]


def _mock_card(
    mocker: pytest_mock.MockerFixture,
    *,
    started: bool = False,
    session: ChatSessionInfo | None = None,
) -> SimpleNamespace:
    session = session or _session()
    return SimpleNamespace(
        session=session,
        validate=mocker.patch.object(
            board_routes,
            "_validate_and_get_session",
            new=AsyncMock(return_value=session),
        ),
        writable=mocker.patch.object(
            board_routes,
            "_validate_and_get_writable_session",
            new=AsyncMock(return_value=session),
        ),
        expert_gate=mocker.patch.object(
            board_routes,
            "_validate_session_expert_writable_by_user",
            new=AsyncMock(),
        ),
        columns=mocker.patch.object(
            board_routes,
            "get_board_columns",
            new=AsyncMock(return_value=_default_columns()),
        ),
        extras=mocker.patch.object(
            board_routes,
            "get_board_session_extras",
            new=AsyncMock(
                return_value={session.session_id: BoardExtras(started=started)}
            ),
        ),
        top=mocker.patch.object(
            board_routes, "top_position", new=AsyncMock(return_value=-1024.0)
        ),
        set_board=mocker.patch.object(
            board_routes, "set_session_board", new=AsyncMock(return_value=True)
        ),
        start=mocker.patch.object(board_routes, "start_board_card", new=AsyncMock()),
    )


# ─── GET /board ───────────────────────────────────────────────────────


def test_get_board_returns_columns_and_sessions(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch.object(
        board_routes,
        "get_board_columns",
        new=AsyncMock(return_value=_default_columns()),
    )
    mock_list = mocker.patch.object(
        board_routes,
        "list_board_sessions",
        new=AsyncMock(return_value=[_row(_session())]),
    )
    mock_experts = mocker.patch.object(
        board_routes.experts_db, "list_experts", new=AsyncMock()
    )

    response = client.get("/board")

    assert response.status_code == 200
    data = response.json()
    assert [c["key"] for c in data["columns"]] == ["backlog", "in_progress", "done"]
    card = data["sessions"][0]
    assert card["id"] == SESSION_ID
    assert card["board_column"] == "in_progress"
    assert card["started"] is True
    assert card["last_line"] == "Drafted the outline."
    assert card["file_count"] == 2
    assert mock_list.await_args.kwargs == {
        "organization_id": "test-org",
        "include_older": False,
    }
    mock_experts.assert_not_awaited()


def test_get_board_overrides_last_line_for_reassigned_cards(
    mocker: pytest_mock.MockerFixture,
) -> None:
    to_priya = _session(
        "sess-priya",
        metadata=ChatSessionMetadata(
            reassigned_to_session_id="sess-new-1", reassigned_to_expert_id="e1"
        ),
    )
    to_otto = _session(
        "sess-otto",
        metadata=ChatSessionMetadata(reassigned_to_session_id="sess-new-2"),
    )
    mocker.patch.object(
        board_routes,
        "get_board_columns",
        new=AsyncMock(return_value=_default_columns()),
    )
    mock_list = mocker.patch.object(
        board_routes,
        "list_board_sessions",
        new=AsyncMock(
            return_value=[
                _row(to_priya, board_column="done"),
                _row(to_otto, board_column="done"),
            ]
        ),
    )
    mocker.patch.object(
        board_routes.experts_db,
        "list_experts",
        new=AsyncMock(return_value=[SimpleNamespace(id="e1", name="Priya")]),
    )

    response = client.get("/board?include_older=true")

    assert response.status_code == 200
    lines = {s["id"]: s["last_line"] for s in response.json()["sessions"]}
    assert lines == {
        "sess-priya": "Reassigned to Priya",
        "sess-otto": "Reassigned to Otto",
    }
    assert mock_list.await_args.kwargs["include_older"] is True


# ─── PUT /board/columns ───────────────────────────────────────────────


def _mock_save_columns(mocker: pytest_mock.MockerFixture) -> AsyncMock:
    return mocker.patch.object(
        board_routes,
        "save_board_columns",
        new=AsyncMock(side_effect=lambda _user_id, columns: validate_columns(columns)),
    )


def test_put_columns_rejects_removed_builtin(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_save_columns(mocker)

    response = client.put(
        "/board/columns",
        json={
            "columns": [
                {"key": "backlog", "name": "Backlog"},
                {"key": "in_progress", "name": "In progress"},
            ]
        },
    )

    assert response.status_code == 422
    assert "done" in response.json()["detail"]


def test_put_columns_returns_saved_columns(
    mocker: pytest_mock.MockerFixture,
) -> None:
    _mock_save_columns(mocker)
    custom_key = "col_6f1c2a4e-1b2c-4d3e-8f90-123456789abc"

    response = client.put(
        "/board/columns",
        json={
            "columns": [
                {"key": "backlog", "name": "Ideas"},
                {"key": custom_key, "name": " Review "},
                {"key": "in_progress", "name": "In progress"},
                {"key": "done", "name": "Done"},
            ]
        },
    )

    assert response.status_code == 200
    columns = response.json()["columns"]
    assert [c["key"] for c in columns] == ["backlog", custom_key, "in_progress", "done"]
    assert columns[1] == {"key": custom_key, "name": "Review", "builtin": False}


# ─── PATCH /sessions/{id}/board ───────────────────────────────────────


def test_patch_board_unknown_column_422(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_card(mocker)

    response = client.patch(
        f"/sessions/{SESSION_ID}/board", json={"column": "col_nope"}
    )

    assert response.status_code == 422
    mocks.set_board.assert_not_awaited()
    mocks.start.assert_not_awaited()


@pytest.mark.parametrize("position", [float("nan"), float("inf")])
def test_patch_board_rejects_non_finite_position(position: float) -> None:
    # Python's JSON parser reads NaN and Infinity; they must not reach the
    # sort key.
    with pytest.raises(pydantic.ValidationError):
        board_routes.UpdateSessionBoardRequest(position=position)


def test_patch_board_missing_session_404(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_card(mocker)
    mocks.validate.side_effect = NotFoundError("Session sess-1 not found.")

    response = client.patch(f"/sessions/{SESSION_ID}/board", json={"column": "done"})

    assert response.status_code == 404
    mocks.set_board.assert_not_awaited()


def test_patch_board_to_in_progress_starts_unstarted_card(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker)

    response = client.patch(
        f"/sessions/{SESSION_ID}/board", json={"column": "in_progress"}
    )

    assert response.status_code == 200
    assert response.json()["id"] == SESSION_ID
    mocks.top.assert_awaited_once_with(TEST_USER_ID, "in_progress")
    mocks.set_board.assert_awaited_once_with(
        SESSION_ID, TEST_USER_ID, column="in_progress", position=-1024.0
    )
    mocks.start.assert_awaited_once_with(
        mocks.session, TEST_USER_ID, org_id="test-org", team_id="test-team"
    )


def test_patch_board_keeps_given_position(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_card(mocker, started=True)

    response = client.patch(
        f"/sessions/{SESSION_ID}/board", json={"column": "done", "position": 3.5}
    )

    assert response.status_code == 200
    mocks.top.assert_not_awaited()
    mocks.set_board.assert_awaited_once_with(
        SESSION_ID, TEST_USER_ID, column="done", position=3.5
    )


def test_patch_board_to_backlog_does_not_start(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker)

    response = client.patch(f"/sessions/{SESSION_ID}/board", json={"column": "backlog"})

    assert response.status_code == 200
    mocks.set_board.assert_awaited_once()
    mocks.start.assert_not_awaited()


def test_patch_board_started_card_does_not_restart(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker, started=True)

    response = client.patch(
        f"/sessions/{SESSION_ID}/board", json={"column": "in_progress"}
    )

    assert response.status_code == 200
    mocks.set_board.assert_awaited_once()
    mocks.start.assert_not_awaited()


def test_patch_board_untitled_card_does_not_start(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker, session=_session(title=None))

    response = client.patch(
        f"/sessions/{SESSION_ID}/board", json={"column": "in_progress"}
    )

    assert response.status_code == 200
    mocks.start.assert_not_awaited()


# ─── PATCH /sessions/{id}/expert ──────────────────────────────────────


def _mock_expert_change(
    mocker: pytest_mock.MockerFixture,
    *,
    result: str = "ok",
    expert: object | None = None,
) -> SimpleNamespace:
    mocks = _mock_card(mocker)
    mocks.get_expert = mocker.patch.object(
        board_routes.experts_db,
        "get_expert",
        new=AsyncMock(
            return_value=expert
            or SimpleNamespace(is_archived=False, schedules_paused_at=None)
        ),
    )
    mocks.tenancy = mocker.patch.object(
        board_routes.experts_db,
        "resolve_private_expert_tenancy",
        new=AsyncMock(return_value=("org-personal", "team-personal")),
    )
    mocks.set_expert = mocker.patch.object(
        board_routes,
        "set_session_expert_if_unstarted",
        new=AsyncMock(return_value=result),
    )
    mocks.invalidate = mocker.patch.object(
        board_routes, "invalidate_session_cache", new=AsyncMock()
    )
    return mocks


def test_patch_expert_started_409(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_expert_change(mocker, result="started")

    response = client.patch(f"/sessions/{SESSION_ID}/expert", json={"expert_id": "e1"})

    assert response.status_code == 409
    assert (
        response.json()["detail"] == "Assignee can only change before the task starts"
    )
    mocks.invalidate.assert_not_awaited()


def test_patch_expert_ok_invalidates_cache(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_expert_change(mocker)

    response = client.patch(f"/sessions/{SESSION_ID}/expert", json={"expert_id": "e1"})

    assert response.status_code == 200
    mocks.set_expert.assert_awaited_once_with(
        SESSION_ID,
        TEST_USER_ID,
        "e1",
        organization_id="org-personal",
        team_id="team-personal",
    )
    mocks.invalidate.assert_awaited_once_with(SESSION_ID)


def test_patch_expert_to_otto_uses_request_tenancy(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_expert_change(mocker)

    response = client.patch(f"/sessions/{SESSION_ID}/expert", json={"expert_id": None})

    assert response.status_code == 200
    mocks.get_expert.assert_not_awaited()
    mocks.set_expert.assert_awaited_once_with(
        SESSION_ID,
        TEST_USER_ID,
        None,
        organization_id="test-org",
        team_id="test-team",
    )
    mocks.invalidate.assert_awaited_once_with(SESSION_ID)


def test_patch_expert_paused_expert_404(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_expert_change(
        mocker,
        expert=SimpleNamespace(
            is_archived=False, schedules_paused_at=datetime.now(UTC)
        ),
    )

    response = client.patch(f"/sessions/{SESSION_ID}/expert", json={"expert_id": "e1"})

    assert response.status_code == 404
    mocks.set_expert.assert_not_awaited()


def test_patch_expert_unknown_session_404(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_expert_change(mocker, result="not_found")

    response = client.patch(f"/sessions/{SESSION_ID}/expert", json={"expert_id": None})

    assert response.status_code == 404
    mocks.invalidate.assert_not_awaited()


# ─── POST /sessions/{id}/start ────────────────────────────────────────


def _mock_turn_dispatch(mocker: pytest_mock.MockerFixture) -> SimpleNamespace:
    mocker.patch.object(server_turn, "enforce_payment_paywall", new=AsyncMock())
    mocker.patch.object(
        server_turn,
        "get_global_rate_limits",
        new=AsyncMock(return_value=(0, 0, SubscriptionTier.BASIC)),
    )
    mocker.patch.object(server_turn, "check_rate_limit", new=AsyncMock())
    return SimpleNamespace(
        schedule=mocker.patch.object(
            server_turn,
            "schedule_chat_turn",
            new=AsyncMock(side_effect=["turn-1", None]),
        ),
        publish=mocker.patch.object(
            server_turn.stream_registry, "publish_chunk", new=AsyncMock()
        ),
    )


def test_start_session_sends_message_once_with_board_start_id(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker)
    mocks.start.side_effect = server_turn.start_board_card
    dispatch = _mock_turn_dispatch(mocker)

    first = client.post(f"/sessions/{SESSION_ID}/start", json={})
    # Pressing Start again before the first message lands is a duplicate.
    second = client.post(f"/sessions/{SESSION_ID}/start", json={})

    assert first.status_code == 200
    assert second.status_code == 200
    expected_id = server_turn.board_start_message_id(TEST_USER_ID, SESSION_ID)
    assert dispatch.schedule.await_count == 2
    for call in dispatch.schedule.await_args_list:
        assert call.kwargs["message_id"] == expected_id
        assert call.kwargs["message"] == "Write the PRD"
        assert call.kwargs["is_user_message"] is True
        assert call.kwargs["organization_id"] == "test-org"
    dispatch.publish.assert_awaited_once()
    mocks.set_board.assert_awaited_with(
        SESSION_ID, TEST_USER_ID, column="in_progress", position=-1024.0
    )


def test_start_session_sends_edited_draft(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_card(mocker)
    mocks.start.side_effect = server_turn.start_board_card
    dispatch = _mock_turn_dispatch(mocker)

    response = client.post(
        f"/sessions/{SESSION_ID}/start", json={"message": "Draft it in Notion"}
    )

    assert response.status_code == 200
    assert dispatch.schedule.await_args.kwargs["message"] == "Draft it in Notion"
    assert response.json()["title"] == "Write the PRD"


def test_start_session_already_started_409(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker, started=True)

    response = client.post(f"/sessions/{SESSION_ID}/start", json={})

    assert response.status_code == 409
    mocks.start.assert_not_awaited()
    mocks.set_board.assert_not_awaited()


def test_start_session_rate_limited_leaves_card_in_place(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker)
    mocks.start.side_effect = fastapi.HTTPException(status_code=429, detail="limit")

    response = client.post(f"/sessions/{SESSION_ID}/start", json={})

    assert response.status_code == 429
    mocks.set_board.assert_not_awaited()


# ─── POST /sessions with a board column ───────────────────────────────


def _mock_create_card(mocker: pytest_mock.MockerFixture) -> SimpleNamespace:
    async def _fake_create(user_id: str, **kwargs):
        return ChatSession.new(user_id, dry_run=False)

    mocker.patch.object(
        chat_routes,
        "_resolve_new_session_llm_route",
        new=AsyncMock(return_value=("platform", None)),
    )
    mocker.patch.object(chat_routes, "enforce_payment_paywall", new=AsyncMock())
    return SimpleNamespace(
        create=mocker.patch.object(
            chat_routes, "create_chat_session", new=AsyncMock(side_effect=_fake_create)
        ),
        columns=mocker.patch.object(
            chat_routes,
            "get_board_columns",
            new=AsyncMock(return_value=_default_columns()),
        ),
        title=mocker.patch.object(
            chat_routes, "update_session_title", new=AsyncMock(return_value=True)
        ),
        top=mocker.patch.object(
            chat_routes, "top_position", new=AsyncMock(return_value=-1024.0)
        ),
        set_board=mocker.patch.object(
            chat_routes, "set_session_board", new=AsyncMock(return_value=True)
        ),
        start=mocker.patch.object(chat_routes, "start_board_card", new=AsyncMock()),
    )


def test_create_session_with_title_and_backlog_is_silent(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_create_card(mocker)

    response = client.post(
        "/sessions", json={"title": "  Write the PRD  ", "board_column": "backlog"}
    )

    assert response.status_code == 200
    session_id = response.json()["id"]
    mocks.title.assert_awaited_once_with(session_id, TEST_USER_ID, "Write the PRD")
    mocks.set_board.assert_awaited_once_with(
        session_id, TEST_USER_ID, column="backlog", position=-1024.0
    )
    mocks.start.assert_not_awaited()


def test_create_session_with_title_and_in_progress_starts(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_create_card(mocker)

    response = client.post(
        "/sessions", json={"title": "Write the PRD", "board_column": "in_progress"}
    )

    assert response.status_code == 200
    mocks.start.assert_awaited_once()
    session, user_id = mocks.start.await_args.args
    assert session.session_id == response.json()["id"]
    assert user_id == TEST_USER_ID
    assert mocks.start.await_args.kwargs == {
        "org_id": "test-org",
        "team_id": "test-team",
        "message": "Write the PRD",
    }


def test_create_session_board_column_with_builder_graph_422(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_create_card(mocker)

    response = client.post(
        "/sessions", json={"board_column": "backlog", "builder_graph_id": "graph-1"}
    )

    assert response.status_code == 422
    mocks.create.assert_not_awaited()
    mocks.set_board.assert_not_awaited()


def test_create_session_unknown_board_column_422(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_create_card(mocker)

    response = client.post(
        "/sessions", json={"title": "Write the PRD", "board_column": "col_nope"}
    )

    assert response.status_code == 422
    mocks.create.assert_not_awaited()


def test_create_session_without_board_fields_skips_board(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_create_card(mocker)

    response = client.post("/sessions", json={})

    assert response.status_code == 200
    mocks.columns.assert_not_awaited()
    mocks.title.assert_not_awaited()
    mocks.set_board.assert_not_awaited()
    mocks.start.assert_not_awaited()
