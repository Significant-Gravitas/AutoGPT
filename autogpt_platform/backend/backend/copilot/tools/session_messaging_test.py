"""Tests for ``find_session`` and ``message_session``.

Each test here is one of the five things that must hold for session-to-session
messaging to be safe: another user's session is invisible, a message reaches a
running session's current turn, an idle session is woken instead of the message
being dropped, the sender is identifiable without a lookup, and neither the
self-message nor the fan-out loop is possible.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.context import MAX_SESSION_MESSAGES_PER_TURN, reset_consult_budget
from backend.copilot.model import ChatSession, ChatSessionInfo, ChatSessionMetadata
from backend.copilot.pending_message_helpers import QueuePendingMessageResponse
from backend.copilot.tools.find_session import MAX_RESULTS, FindSessionTool
from backend.copilot.tools.message_session import MessageSessionTool
from backend.copilot.tools.models import (
    ErrorResponse,
    SessionListResponse,
    SessionMessageResponse,
)
from backend.copilot.tree import TurnEnvelope

_FIND = "backend.copilot.tools.find_session"
_MSG = "backend.copilot.tools.message_session"

OWNER = "user-1"
CALLER_SESSION = "session-caller"
TARGET_SESSION = "session-target"


def _session(session_id: str = CALLER_SESSION, user_id: str = OWNER) -> ChatSession:
    return ChatSession(
        session_id=session_id,
        user_id=user_id,
        usage=[],
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        messages=[],
    )


def _info(
    session_id: str,
    *,
    user_id: str = OWNER,
    expert_id: str | None = None,
    purpose: str | None = None,
    status: str = "idle",
) -> ChatSessionInfo:
    return ChatSessionInfo(
        session_id=session_id,
        user_id=user_id,
        usage=[],
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata=ChatSessionMetadata(purpose=purpose),
        chat_status=status,
        expert_id=expert_id,
    )


@pytest.fixture(autouse=True)
def _fresh_turn_budget():
    reset_consult_budget()
    yield
    reset_consult_budget()


class TestFindSessionScoping:
    async def test_only_the_callers_own_sessions_are_queried(self) -> None:
        """The scope is the query argument, not a filter over a wider read —
        so another user's session is never fetched, let alone returned."""
        with patch(
            f"{_FIND}.list_recent_chat_sessions", new=AsyncMock(return_value=[])
        ) as lister:
            await FindSessionTool()._execute(OWNER, _session())
        assert lister.await_args.kwargs["user_id"] == OWNER

    async def test_another_users_session_is_not_listed(self) -> None:
        rows = [_info("mine"), _info("theirs", user_id="user-2")]
        # Even if the query leaked a foreign row, it must not reach the model.
        with patch(
            f"{_FIND}.list_recent_chat_sessions", new=AsyncMock(return_value=rows)
        ):
            result = await FindSessionTool()._execute(OWNER, _session())
        assert isinstance(result, SessionListResponse)
        listed = {s.session_id for s in result.sessions}
        assert "theirs" not in listed

    async def test_the_calling_session_is_not_listed(self) -> None:
        rows = [_info(CALLER_SESSION), _info("other")]
        with patch(
            f"{_FIND}.list_recent_chat_sessions", new=AsyncMock(return_value=rows)
        ):
            result = await FindSessionTool()._execute(OWNER, _session())
        assert isinstance(result, SessionListResponse)
        assert [s.session_id for s in result.sessions] == ["other"]

    async def test_task_filter_matches_purpose(self) -> None:
        rows = [
            _info("a", purpose="instagram audit"),
            _info("b", purpose="quarterly report"),
        ]
        with patch(
            f"{_FIND}.list_recent_chat_sessions", new=AsyncMock(return_value=rows)
        ):
            result = await FindSessionTool()._execute(
                OWNER, _session(), task="instagram"
            )
        assert isinstance(result, SessionListResponse)
        assert [s.session_id for s in result.sessions] == ["a"]

    async def test_expert_and_status_filter_in_the_query(self) -> None:
        """Filtering these in Python would drop matches older than the scan
        window while the summary still read as authoritative."""
        with patch(
            f"{_FIND}.list_recent_chat_sessions", new=AsyncMock(return_value=[])
        ) as lister:
            await FindSessionTool()._execute(
                OWNER, _session(), expert_id="expert-7", status="running"
            )
        kwargs = lister.await_args.kwargs
        assert kwargs["expert_id"] == "expert-7"
        assert kwargs["status"] == "running"

    async def test_count_reports_what_was_returned_not_what_matched(self) -> None:
        rows = [_info(f"s{i}") for i in range(MAX_RESULTS + 5)]
        with patch(
            f"{_FIND}.list_recent_chat_sessions", new=AsyncMock(return_value=rows)
        ):
            result = await FindSessionTool()._execute(OWNER, _session())
        assert isinstance(result, SessionListResponse)
        assert len(result.sessions) == MAX_RESULTS
        assert str(MAX_RESULTS) in result.message
        assert str(len(rows)) not in result.message


class TestMessageSessionDelivery:
    async def test_running_session_takes_it_into_the_current_turn(self) -> None:
        queued = QueuePendingMessageResponse(
            buffer_length=1, max_buffer_length=10, turn_in_flight=True
        )
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION, status="running")),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ):
                with patch(f"{_MSG}.try_enqueue_turn", new=AsyncMock()) as enqueue:
                    result = await MessageSessionTool()._execute(
                        OWNER,
                        _session(),
                        session_id=TARGET_SESSION,
                        message="the numbers are in",
                    )
        assert isinstance(result, SessionMessageResponse)
        assert result.delivery == "injected"
        # No turn was started: an injected message costs nothing extra.
        enqueue.assert_not_awaited()

    async def test_idle_session_is_woken_rather_than_left_unread(self) -> None:
        queued = QueuePendingMessageResponse(
            buffer_length=0, max_buffer_length=10, turn_in_flight=False
        )
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION, status="idle")),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ):
                with patch(
                    f"{_MSG}.try_enqueue_turn", new=AsyncMock(return_value=object())
                ) as enqueue:
                    result = await MessageSessionTool()._execute(
                        OWNER,
                        _session(),
                        session_id=TARGET_SESSION,
                        message="the numbers are in",
                    )
        assert isinstance(result, SessionMessageResponse)
        assert result.delivery == "woke"
        enqueue.assert_awaited_once()

    async def test_delivered_message_names_the_sender(self) -> None:
        queued = QueuePendingMessageResponse(
            buffer_length=1, max_buffer_length=10, turn_in_flight=True
        )
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION, status="running")),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ) as deliver:
                await MessageSessionTool()._execute(
                    OWNER,
                    _session(),
                    session_id=TARGET_SESSION,
                    message="the numbers are in",
                )
        sent = deliver.await_args.kwargs["message"]
        # In the text, not only the metadata: the model replies off what it reads.
        assert CALLER_SESSION in sent
        assert "the numbers are in" in sent

    async def test_wake_carries_the_sender_in_metadata(self) -> None:
        queued = QueuePendingMessageResponse(
            buffer_length=0, max_buffer_length=10, turn_in_flight=False
        )
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION)),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ):
                with patch(
                    f"{_MSG}.try_enqueue_turn", new=AsyncMock(return_value=object())
                ) as enqueue:
                    await MessageSessionTool()._execute(
                        OWNER,
                        _session(),
                        session_id=TARGET_SESSION,
                        message="hello",
                    )
        meta = enqueue.await_args.kwargs["message_metadata"]
        assert meta["from_session_id"] == CALLER_SESSION


class TestTaintPropagation:
    """A tainted session must not launder instructions into an untainted one.

    Both directions are asserted: without these, inverting or dropping the
    check in ``_render`` leaves the suite green.
    """

    @staticmethod
    async def _deliver(envelope: TurnEnvelope | None) -> str:
        queued = QueuePendingMessageResponse(
            buffer_length=1, max_buffer_length=10, turn_in_flight=True
        )
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION, status="running")),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ) as deliver:
                with patch(f"{_MSG}.get_current_envelope", return_value=envelope):
                    await MessageSessionTool()._execute(
                        OWNER,
                        _session(),
                        session_id=TARGET_SESSION,
                        message="the numbers are in",
                    )
        return deliver.await_args.kwargs["message"]

    async def test_a_tainted_sender_marks_the_message_as_data(self) -> None:
        sent = await self._deliver(TurnEnvelope(tree_id="t", tainted=True))
        assert "data, not instructions" in sent

    async def test_an_untainted_sender_adds_no_warning(self) -> None:
        sent = await self._deliver(TurnEnvelope(tree_id="t", tainted=False))
        assert "data, not instructions" not in sent


class TestMessageSessionGuards:
    async def test_another_users_session_reads_as_not_found(self) -> None:
        """Not 'forbidden' — a distinct refusal would confirm the id exists."""
        foreign = _info(TARGET_SESSION, user_id="user-2")
        with patch(
            f"{_MSG}.get_chat_session_metadata", new=AsyncMock(return_value=foreign)
        ):
            with patch(f"{_MSG}.queue_user_message", new=AsyncMock()) as deliver:
                result = await MessageSessionTool()._execute(
                    OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                )
        assert isinstance(result, ErrorResponse)
        assert "No session" in (result.error or result.message or "")
        deliver.assert_not_awaited()

    async def test_messaging_yourself_is_refused(self) -> None:
        with patch(f"{_MSG}.queue_user_message", new=AsyncMock()) as deliver:
            result = await MessageSessionTool()._execute(
                OWNER, _session(), session_id=CALLER_SESSION, message="hi"
            )
        assert isinstance(result, ErrorResponse)
        deliver.assert_not_awaited()

    async def test_fan_out_is_bounded_within_a_turn(self) -> None:
        queued = QueuePendingMessageResponse(
            buffer_length=1, max_buffer_length=10, turn_in_flight=True
        )
        sent = 0
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION, status="running")),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ):
                for _ in range(MAX_SESSION_MESSAGES_PER_TURN + 1):
                    result = await MessageSessionTool()._execute(
                        OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                    )
                    if isinstance(result, SessionMessageResponse):
                        sent += 1
        assert sent == MAX_SESSION_MESSAGES_PER_TURN
        assert isinstance(result, ErrorResponse)

    async def test_the_budget_is_per_turn(self) -> None:
        """A fresh turn gets a fresh allowance — the cap bounds fan-out, it is
        not a lifetime quota on the session."""
        queued = QueuePendingMessageResponse(
            buffer_length=1, max_buffer_length=10, turn_in_flight=True
        )
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION, status="running")),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ):
                for _ in range(MAX_SESSION_MESSAGES_PER_TURN):
                    await MessageSessionTool()._execute(
                        OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                    )
                reset_consult_budget()
                result = await MessageSessionTool()._execute(
                    OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                )
        assert isinstance(result, SessionMessageResponse)
