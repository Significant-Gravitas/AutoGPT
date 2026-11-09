"""Stop, then a message sent before the stop has settled, on real Redis.

The cancel route and the send route meet in the turn's stream meta and the
follow-up buffer, so both run against Redis here; the executor and the
database are stubbed at their boundaries.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import fastapi
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from httpx import ASGITransport, AsyncClient

from backend.api.features.chat import routes as chat_routes
from backend.copilot import pending_message_helpers, stream_registry
from backend.copilot.active_turns import SessionNotIdle
from backend.copilot.pending_message_helpers import queue_user_message
from backend.copilot.pending_messages import (
    PendingMessage,
    clear_pending_messages_unsafe,
    peek_pending_messages,
    push_pending_message,
)
from backend.data import redis_client
from backend.util.testing import is_tcp_port_reachable

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis reachable; the stream registry needs one to run",
)

app = fastapi.FastAPI()
app.include_router(chat_routes.router)


@pytest.fixture(autouse=True)
def auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def session_id():
    session_id = f"stop-send-{uuid.uuid4().hex}"
    yield session_id
    redis = await redis_client.get_redis_async()
    await redis.delete(stream_registry.get_session_meta_key(session_id))
    await clear_pending_messages_unsafe(session_id)


async def test_a_message_sent_while_a_stop_settles_runs_as_the_next_turn(
    mocker: pytest_mock.MockerFixture, session_id: str, test_user_id: str
) -> None:
    stopped_turn = f"stopped-{uuid.uuid4().hex}"
    await stream_registry.create_session(
        session_id, test_user_id, "chat_stream", "chat", turn_id=stopped_turn
    )
    await push_pending_message(session_id, PendingMessage(content="typed before"))
    world = _stub_boundaries(mocker)

    async def executor_confirms_the_stop() -> None:
        await world.send_reached_gate.wait()
        await stream_registry.mark_session_completed(
            session_id,
            error_message=stream_registry.CANCELLED_MESSAGE,
            skip_error_publish=True,
            turn_id=stopped_turn,
        )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        cancel = asyncio.create_task(client.post(f"/sessions/{session_id}/cancel"))
        await asyncio.wait_for(world.cancel_published.wait(), timeout=10)
        executor = asyncio.create_task(executor_confirms_the_stop())
        send = await asyncio.wait_for(
            client.post(
                f"/sessions/{session_id}/stream",
                json={"message": "do this instead"},
            ),
            timeout=30,
        )
        await asyncio.wait_for(executor, timeout=10)
        cancelled = await asyncio.wait_for(cancel, timeout=10)

    assert cancelled.json()["cancelled"] is True
    assert send.status_code == 200
    assert "data-follow-up-queued" not in send.text
    world.schedule.assert_awaited_once()
    assert world.schedule.await_args.kwargs["message"] == "do this instead"
    assert await peek_pending_messages(session_id) == []


async def test_the_turn_after_a_stopped_one_takes_follow_ups(
    session_id: str, test_user_id: str
) -> None:
    await stream_registry.create_session(
        session_id, test_user_id, "", "", turn_id=f"stopped-{uuid.uuid4().hex}"
    )
    assert await stream_registry.mark_stop_requested(session_id)
    refused = await queue_user_message(
        session_id=session_id, message="too late", require_turn_in_flight=True
    )
    await stream_registry.create_session(
        session_id, test_user_id, "", "", turn_id=f"next-{uuid.uuid4().hex}"
    )
    queued = await queue_user_message(
        session_id=session_id, message="for the next turn", require_turn_in_flight=True
    )

    assert refused.turn_in_flight is False
    assert queued.turn_in_flight is True
    assert [m.content for m in await peek_pending_messages(session_id)] == [
        "for the next turn"
    ]


class _World:
    def __init__(self) -> None:
        self.cancel_published = asyncio.Event()
        self.send_reached_gate = asyncio.Event()
        self.chat_status = "running"
        self.schedule = AsyncMock(side_effect=self._admit)

    async def _admit(self, **kwargs) -> str:
        # The real slot CAS admits a turn only into an idle session.
        if self.chat_status != "idle":
            raise SessionNotIdle(kwargs["session_id"])
        self.chat_status = "running"
        return f"next-{uuid.uuid4().hex}"


def _stub_boundaries(mocker: pytest_mock.MockerFixture) -> _World:
    world = _World()
    session = MagicMock(organization_id=None, team_id=None, expert_id=None)
    session.metadata.llm_auth_provider = "platform"
    session.metadata.llm_credential_id = None

    for name in ("_validate_and_get_session", "_validate_and_get_writable_session"):
        mocker.patch.object(chat_routes, name, new=AsyncMock(return_value=session))
    for name in (
        "enforce_model_entitlement",
        "enforce_payment_paywall",
        "enforce_platform_allowance",
        "record_card_decisions",
        "cancel_background_subtasks",
    ):
        mocker.patch.object(chat_routes, name, new=AsyncMock())
    mocker.patch.object(chat_routes, "spawn_background_task")
    mocker.patch(
        "backend.copilot.briefing.scheduling.ensure_morning_briefing_scheduled",
        new=MagicMock(),
    )
    mocker.patch.object(chat_routes, "cancel_resume_at_reset_in_background")
    mocker.patch.object(chat_routes, "resolve_session_permissions")
    mocker.patch.object(
        chat_routes,
        "background_task_db",
        return_value=MagicMock(
            cancel_queued_background_followup=AsyncMock(return_value=False)
        ),
    )
    mocker.patch.object(
        chat_routes.turn_queue, "cancel_queued_turn", new=AsyncMock(return_value=False)
    )
    mocker.patch.object(chat_routes, "_CANCEL_CONFIRM_POLL_INTERVAL_SECONDS", 0.01)
    mocker.patch.object(chat_routes, "schedule_chat_turn", new=world.schedule)
    mocker.patch.object(
        chat_routes,
        "enqueue_cancel_task",
        new=AsyncMock(side_effect=lambda *_: world.cancel_published.set()),
    )
    mocker.patch.object(
        chat_routes.stream_registry,
        "subscribe_to_session",
        new=AsyncMock(return_value=None),
    )

    real_push = pending_message_helpers.push_pending_message_if_session_running

    async def push_then_signal(*args, **kwargs):
        try:
            return await real_push(*args, **kwargs)
        finally:
            world.send_reached_gate.set()

    mocker.patch.object(
        pending_message_helpers,
        "push_pending_message_if_session_running",
        new=push_then_signal,
    )
    mocker.patch.object(
        pending_message_helpers,
        "chat_db",
        return_value=MagicMock(
            get_chat_session_status=AsyncMock(side_effect=lambda _: world.chat_status)
        ),
    )

    async def release_slot(*_) -> None:
        world.chat_status = "idle"

    mocker.patch.object(stream_registry, "release_turn_slot", new=release_slot)
    mocker.patch.object(stream_registry, "dispatch_next_for_user", new=AsyncMock())
    mocker.patch.object(stream_registry, "publish_chunk", new=AsyncMock())
    mocker.patch.object(stream_registry._notification_bus, "publish", new=AsyncMock())
    return world
