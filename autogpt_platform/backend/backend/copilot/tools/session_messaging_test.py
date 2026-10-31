"""Tests for ``find_session`` and ``message_session``.

Each test here is one of the five things that must hold for session-to-session
messaging to be safe: another user's session is invisible, a message reaches a
running session's current turn, an idle session is woken instead of the message
being dropped, the sender is identifiable without a lookup, and neither the
self-message nor the fan-out loop is possible.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.context import MAX_SESSION_MESSAGES_PER_TURN, reset_consult_budget
from backend.copilot.model import ChatSession, ChatSessionInfo, ChatSessionMetadata
from backend.copilot.pending_message_helpers import QueuePendingMessageResponse
from backend.copilot.session_permissions import BUILDER_BLOCKED_TOOLS
from backend.copilot.tools.find_session import (
    _MAX_SCAN_PAGES,
    _SCAN_LIMIT,
    MAX_RESULTS,
    FindSessionTool,
)
from backend.copilot.tools.message_session import MAX_MESSAGE_CHARS, MessageSessionTool
from backend.copilot.tools.models import (
    ErrorResponse,
    SessionListResponse,
    SessionMessageResponse,
)
from backend.copilot.tree import TurnEnvelope
from backend.copilot.turn_queue import InflightCapExceeded

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
    builder_graph_id: str | None = None,
    llm_auth_provider: CopilotLlmAuthProvider = "platform",
    llm_credential_id: str | None = None,
) -> ChatSessionInfo:
    return ChatSessionInfo(
        session_id=session_id,
        user_id=user_id,
        usage=[],
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata=ChatSessionMetadata(
            purpose=purpose,
            builder_graph_id=builder_graph_id,
            llm_auth_provider=llm_auth_provider,
            llm_credential_id=llm_credential_id,
        ),
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

    async def test_a_match_behind_a_full_page_is_still_found(self) -> None:
        """``task`` is matched in Python, so one page of non-matching rows would
        otherwise hide every older match behind it."""
        pages = [
            [_info(f"new{i}", purpose="unrelated") for i in range(_SCAN_LIMIT)],
            [_info("old", purpose="instagram audit")],
        ]
        lister = AsyncMock(side_effect=pages)
        with patch(f"{_FIND}.list_recent_chat_sessions", new=lister):
            result = await FindSessionTool()._execute(
                OWNER, _session(), task="instagram"
            )
        assert isinstance(result, SessionListResponse)
        assert [s.session_id for s in result.sessions] == ["old"]
        assert lister.await_args_list[1].kwargs["skip"] == _SCAN_LIMIT

    async def test_the_scan_stops_at_the_page_cap(self) -> None:
        """Bounded: an open-ended walk would read a heavy user's whole history
        every time a task matches nothing."""
        page = [_info(f"s{i}", purpose="unrelated") for i in range(_SCAN_LIMIT)]
        lister = AsyncMock(return_value=page)
        with patch(f"{_FIND}.list_recent_chat_sessions", new=lister):
            await FindSessionTool()._execute(OWNER, _session(), task="nothing")
        assert lister.await_count == _MAX_SCAN_PAGES

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
                    f"{_MSG}.get_chat_session_status",
                    new=AsyncMock(return_value="idle"),
                ), patch(
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
                    f"{_MSG}.get_chat_session_status",
                    new=AsyncMock(return_value="idle"),
                ), patch(
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


class TestEveryDeliveryCarriesTheSender:
    """The target's thread renders "Sent from" off the persisted user row,
    so the provenance must ride every delivery path — not only the wake.
    An injected or queued message goes through the pending buffer, whose
    ``PendingMessage`` becomes that row."""

    EXPECTED = {"from_session_id": CALLER_SESSION, "from_expert_id": None}

    async def test_injected_into_a_running_turn(self) -> None:
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
                result = await MessageSessionTool()._execute(
                    OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                )
        assert isinstance(result, SessionMessageResponse)
        assert result.delivery == "injected"
        assert deliver.await_args.kwargs["metadata"] == self.EXPECTED

    async def test_queued_for_a_waiting_turn(self) -> None:
        queued = QueuePendingMessageResponse(
            buffer_length=0, max_buffer_length=10, turn_in_flight=False
        )
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION, status="queued")),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ) as deliver:
                with patch(
                    f"{_MSG}.get_chat_session_status",
                    new=AsyncMock(return_value="queued"),
                ):
                    result = await MessageSessionTool()._execute(
                        OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                    )
        assert isinstance(result, SessionMessageResponse)
        assert result.delivery == "queued"
        assert all(
            call.kwargs["metadata"] == self.EXPECTED for call in deliver.await_args_list
        )

    async def test_woken_from_idle(self) -> None:
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
                    f"{_MSG}.get_chat_session_status",
                    new=AsyncMock(return_value="idle"),
                ), patch(
                    f"{_MSG}.try_enqueue_turn", new=AsyncMock(return_value=object())
                ) as enqueue:
                    result = await MessageSessionTool()._execute(
                        OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                    )
        assert isinstance(result, SessionMessageResponse)
        assert result.delivery == "woke"
        assert enqueue.await_args.kwargs["message_metadata"] == self.EXPECTED

    async def test_an_expert_sender_is_named_by_id(self) -> None:
        queued = QueuePendingMessageResponse(
            buffer_length=1, max_buffer_length=10, turn_in_flight=True
        )
        sender = _session()
        sender.expert_id = "expert-a"
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION, status="running")),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ) as deliver:
                await MessageSessionTool()._execute(
                    OWNER, sender, session_id=TARGET_SESSION, message="hi"
                )
        assert deliver.await_args.kwargs["metadata"] == {
            "from_session_id": CALLER_SESSION,
            "from_expert_id": "expert-a",
        }


class TestWakeCarriesTheTargetsOwnExecutionContext:
    """A woken turn is the TARGET's own turn, so it must run under the
    target's permissions, provider and credential.

    Forwarding none of them dispatches unrestricted, which lifts a builder
    session's tool blocks and re-routes a credential-bound session onto the
    platform default — a widening on a path this tool introduces.
    """

    @staticmethod
    async def _wake_kwargs(target: ChatSessionInfo) -> dict:
        queued = QueuePendingMessageResponse(
            buffer_length=0, max_buffer_length=10, turn_in_flight=False
        )
        with patch(
            f"{_MSG}.get_chat_session_metadata", new=AsyncMock(return_value=target)
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ):
                with patch(
                    f"{_MSG}.get_chat_session_status",
                    new=AsyncMock(return_value="idle"),
                ):
                    with patch(
                        f"{_MSG}.try_enqueue_turn", new=AsyncMock(return_value=object())
                    ) as enqueue:
                        await MessageSessionTool()._execute(
                            OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                        )
        return enqueue.await_args.kwargs

    async def test_builder_bound_target_keeps_its_blocked_tools(self) -> None:
        kwargs = await self._wake_kwargs(
            _info(TARGET_SESSION, builder_graph_id="graph-1")
        )
        perms = kwargs["permissions"]
        assert perms is not None, "a builder session woken unrestricted is a widening"
        assert perms["tools_exclude"] is True
        assert sorted(perms["tools"]) == sorted(BUILDER_BLOCKED_TOOLS)

    async def test_unbound_target_stays_unrestricted(self) -> None:
        kwargs = await self._wake_kwargs(_info(TARGET_SESSION))
        assert kwargs["permissions"] is None

    async def test_target_keeps_its_own_llm_credential(self) -> None:
        kwargs = await self._wake_kwargs(
            _info(
                TARGET_SESSION,
                llm_auth_provider="codex",
                llm_credential_id="cred-9",
            )
        )
        assert kwargs["llm_auth_provider"] == "codex"
        assert kwargs["llm_credential_id"] == "cred-9"


class TestEmptinessIsHonest:
    """``task`` is matched after the scan, so an empty result off a full scan
    means 'not among the recent ones', not 'you have none'."""

    async def test_empty_off_a_full_scan_says_recent_only(self) -> None:
        rows = [_info(f"s{i}", purpose="unrelated") for i in range(_SCAN_LIMIT)]
        with patch(
            f"{_FIND}.list_recent_chat_sessions", new=AsyncMock(return_value=rows)
        ):
            result = await FindSessionTool()._execute(
                OWNER, _session(), task="nothing matches this"
            )
        assert isinstance(result, SessionListResponse)
        assert "recent" in result.message

    async def test_empty_off_a_short_scan_does_not_claim_a_window(self) -> None:
        with patch(
            f"{_FIND}.list_recent_chat_sessions", new=AsyncMock(return_value=[])
        ):
            result = await FindSessionTool()._execute(OWNER, _session(), task="x")
        assert isinstance(result, SessionListResponse)
        assert "recent" not in result.message


class TestMessageLimitsAndCaps:
    async def test_an_overlong_message_is_refused_before_delivery(self) -> None:
        with patch(f"{_MSG}.get_chat_session_metadata", new=AsyncMock()) as fetch:
            with patch(f"{_MSG}.queue_user_message", new=AsyncMock()) as deliver:
                result = await MessageSessionTool()._execute(
                    OWNER,
                    _session(),
                    session_id=TARGET_SESSION,
                    message="x" * (MAX_MESSAGE_CHARS + 1),
                )
        assert isinstance(result, ErrorResponse)
        # Refused on its own length, before the target is even looked up.
        fetch.assert_not_awaited()
        deliver.assert_not_awaited()

    async def test_over_the_inflight_cap_reports_rather_than_raises(self) -> None:
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
                    f"{_MSG}.get_chat_session_status",
                    new=AsyncMock(return_value="idle"),
                ), patch(
                    f"{_MSG}.try_enqueue_turn",
                    new=AsyncMock(side_effect=InflightCapExceeded()),
                ):
                    result = await MessageSessionTool()._execute(
                        OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                    )
        assert isinstance(result, ErrorResponse)


class TestQueuedTargetRidesItsOwnTurn:
    """A queued target must not be woken again.

    ``enqueue_turn`` would append a newer user row, and the dispatcher replays
    a queued turn from the LATEST one — so the user's own submit-time payload
    would be replaced by this message's.
    """

    async def test_queued_target_is_not_enqueued_again(self) -> None:
        queued = QueuePendingMessageResponse(
            buffer_length=0, max_buffer_length=10, turn_in_flight=False
        )
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION, status="queued")),
        ):
            with patch(
                f"{_MSG}.queue_user_message", new=AsyncMock(return_value=queued)
            ) as deliver:
                with patch(
                    f"{_MSG}.get_chat_session_status",
                    new=AsyncMock(return_value="queued"),
                ):
                    with patch(f"{_MSG}.try_enqueue_turn", new=AsyncMock()) as enqueue:
                        result = await MessageSessionTool()._execute(
                            OWNER, _session(), session_id=TARGET_SESSION, message="hi"
                        )
        assert isinstance(result, SessionMessageResponse)
        assert result.delivery == "queued"
        enqueue.assert_not_awaited()
        # The second push is unconditional: the waiting turn drains it.
        assert deliver.await_count == 2


class TestDryRun:
    async def test_dry_run_sends_nothing(self) -> None:
        dry = _session()
        dry.metadata.dry_run = True
        with patch(
            f"{_MSG}.get_chat_session_metadata",
            new=AsyncMock(return_value=_info(TARGET_SESSION)),
        ):
            with patch(f"{_MSG}.queue_user_message", new=AsyncMock()) as deliver:
                with patch(f"{_MSG}.try_enqueue_turn", new=AsyncMock()) as enqueue:
                    result = await MessageSessionTool()._execute(
                        OWNER, dry, session_id=TARGET_SESSION, message="hi"
                    )
        assert isinstance(result, SessionMessageResponse)
        deliver.assert_not_awaited()
        enqueue.assert_not_awaited()


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

    async def test_the_budget_binds_across_the_per_call_task_boundary(self) -> None:
        """The SDK CLI runs every tool call in its own task, which copies the
        context — a count kept as an int and re-``set()`` there never reaches
        the next call, so the cap would bound nothing."""
        reset_consult_budget()
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
                    result = await asyncio.create_task(
                        MessageSessionTool()._execute(
                            OWNER,
                            _session(),
                            session_id=TARGET_SESSION,
                            message="hi",
                        )
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
