"""v2 AutoPilot sessions: a status feed an API key can poll."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from prisma.enums import APIKeyPermission

from backend.api.features.experts.models import ExpertIdentity
from backend.copilot.model import ChatSessionInfo, ChatSessionMetadata

from .autopilot import autopilot_router
from .errors import add_v2_exception_handlers
from .pagination import encode_page_cursor, encode_token_cursor
from .tenancy import TenantContext, require_auth

USER_ID = "user-1"
ORG_ID = "org-1"
_NOW = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)

app = fastapi.FastAPI()
app.include_router(autopilot_router, prefix="/autopilot")
add_v2_exception_handlers(app)
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)


def _auth(scopes: list[APIKeyPermission]) -> None:
    async def fake_auth() -> TenantContext:
        return TenantContext(
            user_id=USER_ID, scopes=scopes, type="api_key", organization_id=ORG_ID
        )

    app.dependency_overrides[require_auth] = fake_auth


@pytest.fixture(autouse=True)
def _scoped_key():
    _auth([APIKeyPermission.READ_AUTOPILOT])
    yield
    app.dependency_overrides.clear()


def _session(
    session_id: str,
    *,
    expert_id: str | None = None,
    chat_status: str = "idle",
    minutes_ago: int = 0,
) -> ChatSessionInfo:
    return ChatSessionInfo(
        session_id=session_id,
        user_id=USER_ID,
        title=f"Chat {session_id}",
        usage=[],
        started_at=_NOW - timedelta(days=1),
        updated_at=_NOW - timedelta(minutes=minutes_ago),
        metadata=ChatSessionMetadata(source_platform="discord" if expert_id else None),
        chat_status=chat_status,
        is_pinned=bool(expert_id),
        expert_id=expert_id,
    )


@pytest.fixture
def sessions(mocker: pytest_mock.MockFixture) -> AsyncMock:
    return mocker.patch(
        "backend.api.external.v2.autopilot.get_user_sessions",
        new_callable=AsyncMock,
        return_value=(
            [
                _session("otto", chat_status="running"),
                _session("expert", expert_id="exp-1", minutes_ago=5),
                _session("gone", expert_id="exp-archived", minutes_ago=9),
            ],
            3,
        ),
    )


@pytest.fixture
def redis_statuses(mocker: pytest_mock.MockFixture) -> MagicMock:
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=["running", None, "completed"])
    redis = MagicMock(pipeline=MagicMock(return_value=pipe))
    mocker.patch(
        "backend.copilot.stream_registry.get_redis_async",
        new_callable=AsyncMock,
        return_value=redis,
    )
    return pipe


@pytest.fixture
def waiting_reviews(mocker: pytest_mock.MockFixture) -> AsyncMock:
    find_many = AsyncMock(
        return_value=[
            Mock(chatSessionId="expert", graphExecId=None),
            # The synthetic-graph shape an older deploy wrote.
            Mock(chatSessionId=None, graphExecId="copilot-session-gone"),
        ]
    )
    mocker.patch(
        "backend.data.human_review.PendingHumanReview.prisma",
        return_value=Mock(find_many=find_many),
    )
    return find_many


@pytest.fixture
def experts(mocker: pytest_mock.MockFixture) -> AsyncMock:
    identities = AsyncMock(
        return_value=[
            ExpertIdentity(
                id="exp-1", name="Ada", avatar_url=None, role="r", is_archived=False
            )
        ]
    )
    mocker.patch(
        "backend.api.external.v2.autopilot.experts_db",
        return_value=Mock(list_expert_identities=identities),
    )
    return identities


def test_lists_sessions_with_live_status(
    sessions: AsyncMock,
    redis_statuses: MagicMock,
    waiting_reviews: AsyncMock,
    experts: AsyncMock,
) -> None:
    response = client.get("/autopilot/sessions")

    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 3
    assert body["next_cursor"] is None
    assert body["items"][0] == {
        "id": "otto",
        "title": "Chat otto",
        "status": "running",
        "is_processing": True,
        "needs_user": False,
        "expert_id": None,
        "expert_name": None,
        "source_platform": None,
        "is_pinned": False,
        "created_at": "2026-10-26T12:00:00Z",
        "updated_at": "2026-10-27T12:00:00Z",
    }
    expert = body["items"][1]
    assert expert["expert_id"] == "exp-1"
    assert expert["expert_name"] == "Ada"
    assert expert["needs_user"] is True
    assert expert["is_processing"] is False
    assert expert["source_platform"] == "discord"
    gone = body["items"][2]
    assert gone["expert_name"] is None
    assert gone["needs_user"] is True
    assert gone["is_processing"] is False

    experts.assert_awaited_once_with(USER_ID)
    where = waiting_reviews.await_args.kwargs["where"]
    assert where["userId"] == USER_ID
    assert where["status"] == "WAITING"


@pytest.fixture
def no_live_status(mocker: pytest_mock.MockFixture) -> None:
    mocker.patch(
        "backend.api.external.v2.autopilot.stream_registry.running_session_ids",
        new_callable=AsyncMock,
        return_value=set(),
    )
    mocker.patch(
        "backend.api.external.v2.autopilot.chat_sessions_awaiting_review",
        new_callable=AsyncMock,
        return_value=set(),
    )
    mocker.patch(
        "backend.api.external.v2.autopilot.experts_db",
        return_value=Mock(list_expert_identities=AsyncMock(return_value=[])),
    )


@pytest.mark.usefixtures("no_live_status")
def test_lists_in_strict_recency_within_the_keys_org(sessions: AsyncMock) -> None:
    client.get("/autopilot/sessions?expert_id=exp-1")

    args, kwargs = sessions.await_args
    assert args == (USER_ID, 10, 0)
    assert kwargs == {
        "organization_id": ORG_ID,
        "expert_id": "exp-1",
        "pinned_first": False,
    }


@pytest.mark.usefixtures("no_live_status")
def test_pages_with_an_opaque_cursor(sessions: AsyncMock) -> None:
    sessions.return_value = (sessions.return_value[0][:2], 5)

    first = client.get("/autopilot/sessions?limit=2").json()
    second = client.get(
        f"/autopilot/sessions?limit=2&cursor={first['next_cursor']}"
    ).json()

    assert first["next_cursor"] == encode_page_cursor(2)
    assert second["next_cursor"] == encode_page_cursor(3)
    assert sessions.await_args.args == (USER_ID, 2, 2)


def test_a_key_without_the_scope_is_refused(sessions: AsyncMock) -> None:
    _auth([APIKeyPermission.READ_RUN, APIKeyPermission.READ_LIBRARY])

    response = client.get("/autopilot/sessions")

    assert response.status_code == 403
    assert "READ_AUTOPILOT" in response.text
    sessions.assert_not_awaited()


@pytest.mark.parametrize("limit", [0, 51])
def test_limit_is_bounded(sessions: AsyncMock, limit: int) -> None:
    response = client.get(f"/autopilot/sessions?limit={limit}")

    assert response.status_code == 422
    sessions.assert_not_awaited()


@pytest.mark.parametrize("cursor", ["garbage", encode_token_cursor("t")])
def test_a_foreign_cursor_is_refused(sessions: AsyncMock, cursor: str) -> None:
    response = client.get(f"/autopilot/sessions?cursor={cursor}")

    assert response.status_code == 400
    sessions.assert_not_awaited()


def test_status_lookups_failing_never_fail_the_list(
    sessions: AsyncMock, mocker: pytest_mock.MockFixture
) -> None:
    mocker.patch(
        "backend.copilot.stream_registry.get_redis_async",
        side_effect=Exception("Redis down"),
    )
    mocker.patch(
        "backend.data.human_review.PendingHumanReview.prisma",
        side_effect=Exception("DB down"),
    )
    mocker.patch(
        "backend.api.external.v2.autopilot.experts_db",
        side_effect=Exception("DB down"),
    )

    response = client.get("/autopilot/sessions")

    assert response.status_code == 200
    items = response.json()["items"]
    assert [i["is_processing"] for i in items] == [False, False, False]
    assert [i["needs_user"] for i in items] == [False, False, False]
    assert [i["expert_name"] for i in items] == [None, None, None]
