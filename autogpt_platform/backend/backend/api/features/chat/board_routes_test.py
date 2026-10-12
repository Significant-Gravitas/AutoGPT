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
from backend.copilot.board import DEFAULT_COLUMNS, REASSIGN_BRIEF_KIND, validate_columns
from backend.copilot.board_db import BoardExtras, BoardSessionRow
from backend.copilot.expert_kickoff import scoped_client_message_id
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    ChatSessionInfo,
    ChatSessionMetadata,
)
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
        chat_routes, "processing_session_ids", new=AsyncMock(return_value=set())
    )
    yield
    app.dependency_overrides.clear()


def _session(
    session_id: str = SESSION_ID,
    *,
    title: str | None = "Write the PRD",
    expert_id: str | None = None,
    metadata: ChatSessionMetadata | None = None,
    chat_status: str = "idle",
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
        chat_status=chat_status,
    )


def _row(session: ChatSessionInfo, **overrides) -> BoardSessionRow:
    fields = {
        "board_column": "in_progress",
        "board_position": 0.0,
        "started": True,
        "has_messages": True,
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
    assert card["has_messages"] is True
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
    reassigned_to = {
        s["id"]: s["reassigned_to_session_id"] for s in response.json()["sessions"]
    }
    assert reassigned_to == {"sess-priya": "sess-new-1", "sess-otto": "sess-new-2"}
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


def test_patch_board_refused_start_leaves_card_in_place(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker)
    mocks.start.side_effect = fastapi.HTTPException(
        status_code=402, detail="Add credits to keep going"
    )

    response = client.patch(
        f"/sessions/{SESSION_ID}/board", json={"column": "in_progress"}
    )

    assert response.status_code == 402
    assert response.json()["detail"] == "Add credits to keep going"
    mocks.set_board.assert_not_awaited()


def test_patch_board_reassigned_card_says_who_took_it(
    mocker: pytest_mock.MockerFixture,
) -> None:
    reassigned = _session(
        metadata=ChatSessionMetadata(
            reassigned_to_session_id="sess-new", reassigned_to_expert_id="e1"
        ),
    )
    mocks = _mock_card(mocker, started=True, session=reassigned)
    mocks.extras.return_value = {
        SESSION_ID: BoardExtras(started=True, last_line="Drafted the outline.")
    }
    mocker.patch.object(
        board_routes.experts_db,
        "list_experts",
        new=AsyncMock(return_value=[SimpleNamespace(id="e1", name="Priya")]),
    )

    response = client.patch(f"/sessions/{SESSION_ID}/board", json={"column": "done"})

    assert response.status_code == 200
    assert response.json()["last_line"] == "Reassigned to Priya"


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


def test_patch_board_to_done_does_not_start(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker)

    response = client.patch(f"/sessions/{SESSION_ID}/board", json={"column": "done"})

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


def test_patch_board_running_unstarted_card_moves_without_starting(
    mocker: pytest_mock.MockerFixture,
) -> None:
    # Only hidden rows (an expert kickoff) so far, but a turn is running.
    mocks = _mock_card(mocker, session=_session(chat_status="running"))

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
        in_flight=mocker.patch.object(
            server_turn, "is_turn_in_flight", new=AsyncMock(return_value=False)
        ),
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


def test_start_session_running_409(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_card(mocker, session=_session(chat_status="running"))

    response = client.post(f"/sessions/{SESSION_ID}/start", json={})

    assert response.status_code == 409
    mocks.start.assert_not_awaited()
    mocks.set_board.assert_not_awaited()


async def test_dispatch_server_turn_refuses_while_a_turn_runs(
    mocker: pytest_mock.MockerFixture,
) -> None:
    dispatch = _mock_turn_dispatch(mocker)
    dispatch.in_flight.return_value = True

    with pytest.raises(fastapi.HTTPException) as exc:
        await server_turn.dispatch_server_turn(
            _session(),
            TEST_USER_ID,
            org_id="test-org",
            team_id="test-team",
            message="Write the PRD",
            message_id="m-1",
        )

    assert exc.value.status_code == 409
    dispatch.schedule.assert_not_awaited()


def test_start_session_rate_limited_leaves_card_in_place(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_card(mocker)
    mocks.start.side_effect = fastapi.HTTPException(status_code=429, detail="limit")

    response = client.post(f"/sessions/{SESSION_ID}/start", json={})

    assert response.status_code == 429
    mocks.set_board.assert_not_awaited()


# ─── POST /sessions/{id}/reassign ─────────────────────────────────────

NEW_SESSION_ID = "sess-new"


def _expert(name: str, **overrides) -> SimpleNamespace:
    fields = {"name": name, "is_archived": False, "schedules_paused_at": None}
    return SimpleNamespace(**{**fields, **overrides})


def _mock_reassign(
    mocker: pytest_mock.MockerFixture,
    *,
    old: ChatSessionInfo | None = None,
    started: bool = True,
    in_flight: bool = False,
    target: SimpleNamespace | None = None,
) -> SimpleNamespace:
    old = old or _session(
        expert_id="e-old",
        metadata=ChatSessionMetadata(llm_auth_provider="platform", dry_run=True),
    )
    experts = {"e-old": _expert("Nova"), "e-new": target or _expert("Priya")}
    created: list[ChatSession] = []

    async def _fake_create(user_id: str, **kwargs):
        session = ChatSession.new(user_id, session_id=NEW_SESSION_ID, **kwargs)
        created.append(session)
        return session

    async def _fake_validate(session_id: str, _user_id: str):
        return old if session_id == old.session_id else created[-1]

    async def _fake_mark(
        _session_id: str,
        _user_id: str,
        *,
        to_session_id: str,
        to_expert_id: str | None,
    ) -> None:
        old.metadata.reassigned_to_session_id = to_session_id
        old.metadata.reassigned_to_expert_id = to_expert_id

    full_old = ChatSession.new(TEST_USER_ID, dry_run=True, session_id=SESSION_ID)
    full_old.messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]
    return SimpleNamespace(
        old=old,
        created=created,
        writable=mocker.patch.object(
            board_routes,
            "_validate_and_get_writable_session",
            new=AsyncMock(return_value=old),
        ),
        validate=mocker.patch.object(
            board_routes,
            "_validate_and_get_session",
            new=AsyncMock(side_effect=_fake_validate),
        ),
        in_flight=mocker.patch.object(
            server_turn, "is_turn_in_flight", new=AsyncMock(return_value=in_flight)
        ),
        extras=mocker.patch.object(
            board_routes,
            "get_board_session_extras",
            new=AsyncMock(
                side_effect=lambda _user_id, ids: {
                    sid: BoardExtras(started=started) for sid in ids
                }
            ),
        ),
        get_expert=mocker.patch.object(
            board_routes.experts_db,
            "get_expert",
            new=AsyncMock(side_effect=lambda _user_id, eid, **_: experts.get(eid)),
        ),
        list_experts=mocker.patch.object(
            board_routes.experts_db,
            "list_experts",
            new=AsyncMock(
                side_effect=lambda _user_id, **_: [
                    SimpleNamespace(id=eid, name=expert.name)
                    for eid, expert in experts.items()
                ]
            ),
        ),
        create=mocker.patch.object(
            board_routes, "create_chat_session", new=AsyncMock(side_effect=_fake_create)
        ),
        title=mocker.patch.object(
            board_routes, "update_session_title", new=AsyncMock(return_value=True)
        ),
        top=mocker.patch.object(
            board_routes, "top_position", new=AsyncMock(return_value=-1024.0)
        ),
        set_board=mocker.patch.object(
            board_routes, "set_session_board", new=AsyncMock(return_value=True)
        ),
        history=mocker.patch.object(
            board_routes, "get_chat_session", new=AsyncMock(return_value=full_old)
        ),
        files=mocker.patch.object(
            board_routes,
            "list_session_files",
            new=AsyncMock(return_value=[("plan.md", "f-1")]),
        ),
        dispatch=mocker.patch.object(
            board_routes, "dispatch_server_turn", new=AsyncMock(return_value="turn-1")
        ),
        mark=mocker.patch.object(
            board_routes,
            "mark_session_reassigned",
            new=AsyncMock(side_effect=_fake_mark),
        ),
        invalidate=mocker.patch.object(
            board_routes, "invalidate_session_cache", new=AsyncMock()
        ),
        delete=mocker.patch.object(
            board_routes, "delete_chat_session", new=AsyncMock(return_value=True)
        ),
    )


def test_reassign_opens_a_new_session_briefed_by_the_old_one(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_reassign(mocker)

    response = client.post(
        f"/sessions/{SESSION_ID}/reassign", json={"expert_id": "e-new"}
    )

    assert response.status_code == 200
    mocks.create.assert_awaited_once_with(
        TEST_USER_ID,
        dry_run=True,
        organization_id="test-org",
        team_id="test-team",
        llm_auth_provider="platform",
        llm_credential_id=None,
        expert_id="e-new",
        kind="reassigned",
        reassigned_from_session_id=SESSION_ID,
        reassigned_from_expert_id="e-old",
    )
    mocks.title.assert_awaited_once_with(NEW_SESSION_ID, TEST_USER_ID, "Write the PRD")
    mocks.set_board.assert_awaited_once_with(
        NEW_SESSION_ID, TEST_USER_ID, column="in_progress", position=-1024.0
    )
    mocks.history.assert_awaited_once_with(SESSION_ID, TEST_USER_ID)
    mocks.files.assert_awaited_once_with(TEST_USER_ID, SESSION_ID)

    mocks.dispatch.assert_awaited_once()
    session, user_id = mocks.dispatch.await_args.args
    kwargs = mocks.dispatch.await_args.kwargs
    assert session.session_id == NEW_SESSION_ID
    assert user_id == TEST_USER_ID
    assert kwargs["org_id"] == "test-org"
    assert kwargs["team_id"] == "test-team"
    assert kwargs["message_id"] == scoped_client_message_id(
        TEST_USER_ID, NEW_SESSION_ID, "reassign-brief"
    )
    assert kwargs["message_metadata"] == {
        "hidden": True,
        "kind": REASSIGN_BRIEF_KIND,
        "expert_id": "e-new",
    }
    assert kwargs["message"].startswith(
        "The user reassigned this task from Nova to you."
    )
    assert "User: hi\nNova: hello" in kwargs["message"]
    assert "- plan.md (file_id: f-1)" in kwargs["message"]

    mocks.mark.assert_awaited_once_with(
        SESSION_ID, TEST_USER_ID, to_session_id=NEW_SESSION_ID, to_expert_id="e-new"
    )
    mocks.invalidate.assert_awaited_once_with(SESSION_ID)
    mocks.delete.assert_not_awaited()
    data = response.json()
    assert data["old_session"]["id"] == SESSION_ID
    assert data["old_session"]["last_line"] == "Reassigned to Priya"
    assert data["new_session"]["id"] == NEW_SESSION_ID


def test_reassign_to_otto_names_otto(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_reassign(mocker)

    response = client.post(f"/sessions/{SESSION_ID}/reassign", json={"expert_id": None})

    assert response.status_code == 200
    assert mocks.create.await_args.kwargs["expert_id"] is None
    assert mocks.dispatch.await_args.kwargs["message_metadata"]["expert_id"] is None
    assert response.json()["old_session"]["last_line"] == "Reassigned to Otto"


@pytest.mark.parametrize(
    "case",
    ["running", "in_flight", "unstarted", "already_reassigned"],
)
def test_reassign_refuses_a_card_it_cannot_take_409(
    mocker: pytest_mock.MockerFixture, case: str
) -> None:
    old = _session(
        expert_id="e-old",
        chat_status="running" if case == "running" else "idle",
        metadata=ChatSessionMetadata(
            reassigned_to_session_id=(
                "sess-elsewhere" if case == "already_reassigned" else None
            )
        ),
    )
    mocks = _mock_reassign(
        mocker,
        old=old,
        in_flight=case == "in_flight",
        started=case != "unstarted",
    )

    response = client.post(
        f"/sessions/{SESSION_ID}/reassign", json={"expert_id": "e-new"}
    )

    assert response.status_code == 409
    mocks.create.assert_not_awaited()
    mocks.dispatch.assert_not_awaited()
    mocks.mark.assert_not_awaited()


def test_reassign_to_the_same_assignee_422(mocker: pytest_mock.MockerFixture) -> None:
    mocks = _mock_reassign(mocker)

    response = client.post(
        f"/sessions/{SESSION_ID}/reassign", json={"expert_id": "e-old"}
    )

    assert response.status_code == 422
    mocks.create.assert_not_awaited()


@pytest.mark.parametrize(
    "target",
    [
        _expert("Priya", schedules_paused_at=datetime.now(UTC)),
        _expert("Priya", is_archived=True),
        None,
    ],
    ids=["paused", "archived", "missing"],
)
def test_reassign_to_an_unavailable_expert_404(
    mocker: pytest_mock.MockerFixture, target: SimpleNamespace | None
) -> None:
    mocks = _mock_reassign(mocker, target=target)
    if target is None:
        mocks.get_expert.side_effect = lambda _user_id, eid, **_: (
            _expert("Nova") if eid == "e-old" else None
        )

    response = client.post(
        f"/sessions/{SESSION_ID}/reassign", json={"expert_id": "e-new"}
    )

    assert response.status_code == 404
    mocks.create.assert_not_awaited()


def test_reassign_refused_turn_deletes_the_new_session(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_reassign(mocker)
    mocks.dispatch.side_effect = fastapi.HTTPException(status_code=429, detail="limit")

    response = client.post(
        f"/sessions/{SESSION_ID}/reassign", json={"expert_id": "e-new"}
    )

    assert response.status_code == 429
    mocks.delete.assert_awaited_once_with(NEW_SESSION_ID, TEST_USER_ID)
    mocks.mark.assert_not_awaited()
    mocks.invalidate.assert_not_awaited()


def test_reassign_failed_cleanup_keeps_the_original_error(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_reassign(mocker)
    mocks.dispatch.side_effect = fastapi.HTTPException(status_code=429, detail="limit")
    mocks.delete.side_effect = RuntimeError("db down")

    response = client.post(
        f"/sessions/{SESSION_ID}/reassign", json={"expert_id": "e-new"}
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "limit"
    mocks.delete.assert_awaited_once_with(NEW_SESSION_ID, TEST_USER_ID)
    mocks.mark.assert_not_awaited()


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
        rate_limit=mocker.patch.object(
            chat_routes, "check_platform_rate_limit", new=AsyncMock()
        ),
        delete=mocker.patch.object(
            chat_routes, "delete_chat_session", new=AsyncMock(return_value=True)
        ),
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
    mocks.rate_limit.assert_not_awaited()


def test_create_session_with_title_and_done_is_silent(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_create_card(mocker)

    response = client.post(
        "/sessions", json={"title": "Write the PRD", "board_column": "done"}
    )

    assert response.status_code == 200
    mocks.set_board.assert_awaited_once()
    mocks.start.assert_not_awaited()
    mocks.rate_limit.assert_not_awaited()


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
    mocks.rate_limit.assert_awaited_once_with(TEST_USER_ID)
    mocks.delete.assert_not_awaited()


def test_create_session_over_the_rate_limit_creates_nothing(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_create_card(mocker)
    mocks.rate_limit.side_effect = fastapi.HTTPException(
        status_code=429, detail="You've reached your daily limit"
    )

    response = client.post(
        "/sessions", json={"title": "Write the PRD", "board_column": "in_progress"}
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "You've reached your daily limit"
    mocks.create.assert_not_awaited()
    mocks.start.assert_not_awaited()


def test_create_session_refused_start_deletes_the_new_session(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_create_card(mocker)
    mocks.start.side_effect = fastapi.HTTPException(status_code=429, detail="limit")

    response = client.post(
        "/sessions", json={"title": "Write the PRD", "board_column": "in_progress"}
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "limit"
    session, _ = mocks.start.await_args.args
    mocks.delete.assert_awaited_once_with(session.session_id, TEST_USER_ID)


def test_create_session_failed_cleanup_keeps_the_original_error(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocks = _mock_create_card(mocker)
    mocks.start.side_effect = fastapi.HTTPException(status_code=429, detail="limit")
    mocks.delete.side_effect = RuntimeError("db down")

    response = client.post(
        "/sessions", json={"title": "Write the PRD", "board_column": "in_progress"}
    )

    assert response.status_code == 429
    assert response.json()["detail"] == "limit"
    mocks.delete.assert_awaited_once()


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


# ─── GET /sessions board facts ────────────────────────────────────────


def _mock_session_list(mocker: pytest_mock.MockerFixture) -> AsyncMock:
    mocker.patch.object(
        chat_routes,
        "get_user_sessions",
        new=AsyncMock(return_value=([_session()], 1)),
    )
    return mocker.patch.object(
        chat_routes,
        "get_board_session_extras",
        new=AsyncMock(
            return_value={
                SESSION_ID: BoardExtras(
                    board_column="backlog", started=False, has_messages=True
                )
            }
        ),
    )


def test_list_sessions_skips_board_facts_by_default(
    mocker: pytest_mock.MockerFixture,
) -> None:
    extras = _mock_session_list(mocker)

    response = client.get("/sessions")

    assert response.status_code == 200
    assert response.json()["sessions"][0]["board_column"] is None
    extras.assert_not_awaited()


def test_list_sessions_adds_board_facts_on_request(
    mocker: pytest_mock.MockerFixture,
) -> None:
    extras = _mock_session_list(mocker)

    response = client.get("/sessions?include_board=true")

    assert response.status_code == 200
    card = response.json()["sessions"][0]
    assert card["board_column"] == "backlog"
    assert card["has_messages"] is True
    extras.assert_awaited_once_with(TEST_USER_ID, [SESSION_ID])


def test_list_sessions_falls_back_when_board_facts_fail(
    mocker: pytest_mock.MockerFixture,
) -> None:
    extras = _mock_session_list(mocker)
    extras.side_effect = RuntimeError("db down")

    response = client.get("/sessions?include_board=true")

    assert response.status_code == 200
    card = response.json()["sessions"][0]
    assert card["id"] == SESSION_ID
    assert card["board_column"] is None
    assert card["started"] is False
