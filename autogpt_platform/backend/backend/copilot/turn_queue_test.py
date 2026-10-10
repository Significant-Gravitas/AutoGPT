"""Unit tests for turn_queue: per-user FIFO queue layered over
ChatSession.chatStatus.

DB access is mocked via the ``backend.copilot.turn_queue.chat_db``
indirection — same accessor pattern the executor subprocess uses to
RPC into ``DatabaseManager``. Patching the accessor avoids reaching
for Prisma directly while still exercising the queue's branching.
"""

import asyncio
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.errors import UniqueViolationError

from backend.copilot import turn_queue
from backend.copilot.gate import held
from backend.copilot.model import ChatMessage as PydanticChatMessage
from backend.copilot.model import ChatSessionInfo
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.tree import TreeRefusal, TurnEnvelope


class _NoopAsyncCM:
    """Stand-in for the Redis NX session lock context manager."""

    async def __aenter__(self):
        return True

    async def __aexit__(self, *exc):
        return None


def _pyd_message(**overrides) -> PydanticChatMessage:
    """Build a Pydantic ChatMessage with sensible defaults."""
    base = {
        "id": "msg-1",
        "role": "user",
        "content": "hello",
        "session_id": "s1",
        "metadata": None,
        "created_at": datetime.now(timezone.utc),
        "sequence": 1,
    }
    base.update(overrides)
    return PydanticChatMessage(**base)


def _queued_row(session_id: str = "s1", title: str | None = "T") -> ChatSessionInfo:
    """A row as ``list_chat_sessions_by_status`` returns it to the dispatcher."""
    now = datetime.now(timezone.utc)
    return ChatSessionInfo(
        session_id=session_id,
        user_id="u1",
        title=title,
        usage=[],
        started_at=now,
        updated_at=now,
    )


@pytest.fixture(autouse=True)
def tracked_message(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    tracker = MagicMock()
    monkeypatch.setattr(turn_queue, "track_user_message", tracker)
    return tracker


# ── enqueue_turn payload encoding ──────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_turn_packs_metadata_into_metadata_payload() -> None:
    """Non-message dispatch params (file_ids, model, permissions,
    context, request_arrival_at) land in the ChatMessage row's
    ``metadata`` JSONB so the dispatcher can replay the original turn
    shape later."""
    db = MagicMock()
    db.get_next_sequence = AsyncMock(return_value=42)
    db.add_chat_message = AsyncMock(return_value=_pyd_message(sequence=42))
    db.update_chat_session_status = AsyncMock(return_value=True)
    with (
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "_get_session_lock", return_value=_NoopAsyncCM()),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
    ):
        await turn_queue.enqueue_turn(
            user_id="u1",
            session_id="s1",
            message="hello",
            message_id="msg-1",
            message_metadata={"hidden": True, "kind": "expert_kickoff"},
            context={"url": "https://example.com"},
            file_ids=["f1", "f2"],
            model="advanced",
            permissions={"tool_filter": "allow"},
            request_arrival_at=123.45,
        )
    kwargs = db.add_chat_message.call_args.kwargs
    assert kwargs["session_id"] == "s1"
    assert kwargs["sequence"] == 42
    metadata = kwargs["metadata"]
    assert metadata["context"] == {"url": "https://example.com"}
    assert metadata["file_ids"] == ["f1", "f2"]
    assert "mode" not in metadata
    assert metadata["model"] == "advanced"
    assert metadata["llm_auth_provider"] == "platform"
    assert metadata["permissions"] == {"tool_filter": "allow"}
    assert metadata["request_arrival_at"] == 123.45
    assert metadata["hidden"] is True
    assert metadata["kind"] == "expert_kickoff"
    # Session is flipped idle → queued.
    db.update_chat_session_status.assert_awaited_once_with(
        session_id="s1", expect_status="idle", status="queued", user_id="u1"
    )


@pytest.mark.asyncio
async def test_enqueue_turn_only_includes_default_transport_without_extra_params() -> (
    None
):
    """A turn with no extra params only persists its default LLM transport."""
    db = MagicMock()
    db.get_next_sequence = AsyncMock(return_value=1)
    db.add_chat_message = AsyncMock(return_value=_pyd_message())
    db.update_chat_session_status = AsyncMock(return_value=True)
    with (
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "_get_session_lock", return_value=_NoopAsyncCM()),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
    ):
        await turn_queue.enqueue_turn(user_id="u1", session_id="s1", message="hello")
    assert db.add_chat_message.call_args.kwargs["metadata"] == {
        "llm_auth_provider": "platform"
    }


@pytest.mark.asyncio
async def test_enqueue_turn_treats_duplicate_message_pk_as_already_queued() -> None:
    db = MagicMock()
    db.get_next_sequence = AsyncMock(return_value=1)
    db.add_chat_message = AsyncMock(
        side_effect=UniqueViolationError(
            {"user_facing_error": {"message": "ChatMessage_pkey"}}
        )
    )
    db.update_chat_session_status = AsyncMock()
    invalidate = AsyncMock()
    with (
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "_get_session_lock", return_value=_NoopAsyncCM()),
        patch.object(turn_queue, "invalidate_session_cache", new=invalidate),
    ):
        result = await turn_queue.enqueue_turn(
            user_id="u1",
            session_id="s1",
            message="kickoff",
            message_id="same-id",
        )

    assert result is None
    db.update_chat_session_status.assert_not_awaited()
    invalidate.assert_not_awaited()


# ── cancel_queued_turn ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_queued_turn_returns_true_and_invalidates_cache() -> None:
    """A successful cancel flips the session ``queued`` → ``idle`` and
    invalidates the session cache so the frontend drops the badge."""
    db = MagicMock()
    db.update_chat_session_status = AsyncMock(return_value=True)
    invalidate = AsyncMock()
    with (
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "invalidate_session_cache", new=invalidate),
    ):
        ok = await turn_queue.cancel_queued_turn(user_id="u1", session_id="s1")
    assert ok is True
    invalidate.assert_awaited_once_with("s1")
    db.update_chat_session_status.assert_awaited_once_with(
        session_id="s1",
        expect_status="queued",
        status="idle",
        user_id="u1",
    )


@pytest.mark.asyncio
async def test_cancel_queued_turn_returns_false_when_not_owned_or_not_queued() -> None:
    db = MagicMock()
    db.update_chat_session_status = AsyncMock(return_value=False)
    with patch.object(turn_queue, "chat_db", return_value=db):
        ok = await turn_queue.cancel_queued_turn(user_id="u1", session_id="s1")
    assert ok is False


# ── try_enqueue_turn ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_try_enqueue_turn_raises_when_at_inflight_cap() -> None:
    """Pre-check rejects when running + queued already equals the cap."""
    db = MagicMock()
    db.count_chat_sessions_by_status = AsyncMock(return_value=10)
    db.add_chat_message = AsyncMock()
    with (
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "count_running_turns", new=AsyncMock(return_value=5)),
    ):
        with pytest.raises(turn_queue.InflightCapExceeded):
            await turn_queue.try_enqueue_turn(
                user_id="u1",
                inflight_cap=15,
                session_id="s1",
                message="hi",
            )
    db.add_chat_message.assert_not_awaited()


# ── dispatch_next_for_user ─────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "expert_id",
        "origin",
        "source_platform",
        "role",
        "claim",
        "dispatch_fails",
        "tracking_fails",
        "expected_event",
    ),
    [
        ("expert-1", "interactive", None, "user", True, False, False, True),
        (None, "interactive", "discord", "user", True, False, False, True),
        (None, "automation", None, "user", True, False, False, True),
        ("expert-1", "interactive", None, "user", False, False, False, False),
        ("expert-1", "interactive", None, "user", True, True, False, False),
        ("expert-1", "interactive", None, "assistant", True, False, False, False),
        ("expert-1", "interactive", None, "user", True, False, True, True),
    ],
)
async def test_promoted_turn_tracking_preserves_session_attribution(
    tracked_message: MagicMock,
    expert_id: str | None,
    origin: str,
    source_platform: str | None,
    role: str,
    claim: bool,
    dispatch_fails: bool,
    tracking_fails: bool,
    expected_event: bool,
) -> None:
    if tracking_fails:
        tracked_message.side_effect = RuntimeError("tracking failed")
    head = _queued_row()
    head.expert_id = expert_id
    head.metadata.origin = origin
    head.metadata.source_platform = source_platform
    head.metadata.llm_auth_provider = "codex"
    pending = _pyd_message(role=role)
    db = MagicMock()
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    db.update_chat_session_status = AsyncMock(return_value=True)
    dispatched = AsyncMock(
        side_effect=RuntimeError("dispatch failed") if dispatch_fails else None
    )
    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=AsyncMock(return_value=True)),
        patch.object(
            turn_queue,
            "claim_queued_session",
            new=AsyncMock(return_value="admitted" if claim else "full"),
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=dispatched),
    ):
        if dispatch_fails:
            with pytest.raises(RuntimeError, match="dispatch failed"):
                await turn_queue.dispatch_next_for_user("u1")
        else:
            assert await turn_queue.dispatch_next_for_user("u1") is claim

    if expected_event:
        tracked_message.assert_called_once_with(
            user_id="u1",
            session_id="s1",
            message_length=5,
            expert_id=expert_id,
            origin=origin,
            source_platform=source_platform,
        )
    else:
        tracked_message.assert_not_called()


def _patch_queued_list(rows):
    """Patch ``list_queued_sessions`` (the dispatcher's queue read) to
    return the given rows.  Patching the helper rather than the
    underlying RPC keeps the test independent of how chat_db()
    resolves in-process vs. via DatabaseManagerAsyncClient."""
    return patch.object(
        turn_queue, "list_queued_sessions", new=AsyncMock(return_value=rows)
    )


@pytest.mark.asyncio
async def test_a_long_run_of_wakes_that_may_not_start_drains_without_recursing() -> (
    None
):
    """Each is closed and the next tried; more of them than the recursion limit
    must still drain rather than overflow inside the completion hook."""
    queue = [_queued_row(f"s{i}") for i in range(sys.getrecursionlimit() + 100)]
    for row in queue:
        row.metadata.llm_auth_provider = "codex"
    db = MagicMock()
    db.get_latest_user_message_in_session = AsyncMock(
        return_value=_pyd_message(metadata={held._WAKE_KEY: True})
    )

    async def closed(head, reason):
        queue.remove(head)

    with (
        patch.object(
            turn_queue,
            "list_queued_sessions",
            new=AsyncMock(side_effect=lambda _user: list(queue)),
        ),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=AsyncMock(return_value=True)),
        patch.object(
            turn_queue, "claim_queued_session", new=AsyncMock(return_value="admitted")
        ),
        patch.object(turn_queue, "_refuse_queued_turn", new=closed),
    ):
        assert await turn_queue.dispatch_next_for_user("u1") is False

    assert queue == []


@pytest.mark.asyncio
async def test_a_head_taken_since_it_was_listed_does_not_stop_promotion() -> None:
    """Cancelled or claimed elsewhere between the listing and the claim: the
    next queued session is tried rather than the dispatch giving up."""
    gone, waiting = _queued_row("gone"), _queued_row("waiting")
    for row in (gone, waiting):
        row.metadata.llm_auth_provider = "codex"
    db = MagicMock()
    db.get_latest_user_message_in_session = AsyncMock(return_value=_pyd_message())
    dispatched = AsyncMock()

    with (
        patch.object(
            turn_queue,
            "list_queued_sessions",
            new=AsyncMock(side_effect=[[gone, waiting], [waiting]]),
        ),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=AsyncMock(return_value=True)),
        patch.object(
            turn_queue,
            "claim_queued_session",
            new=AsyncMock(side_effect=["busy", "admitted"]),
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=dispatched),
    ):
        assert await turn_queue.dispatch_next_for_user("u1") is True

    assert dispatched.await_args.kwargs["session_id"] == "waiting"


@pytest.mark.asyncio
async def test_unparseable_stored_permissions_read_the_sessions_current_ones() -> None:
    """Not left stuck at the head of the queue, and never read as none."""
    head = _queued_row()
    head.metadata.llm_auth_provider = "codex"
    db = MagicMock()
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(
        return_value=_pyd_message(metadata={"permissions": {"tools": 5}})
    )
    today = CopilotPermissions(tools=["web_fetch"], tools_exclude=True)
    dispatched = AsyncMock()

    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=AsyncMock(return_value=True)),
        patch.object(
            turn_queue, "claim_queued_session", new=AsyncMock(return_value="admitted")
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch.object(turn_queue, "resolve_session_permissions", return_value=today),
        patch("backend.copilot.executor.utils.dispatch_turn", new=dispatched),
    ):
        assert await turn_queue.dispatch_next_for_user("u1") is True

    assert dispatched.await_args.kwargs["permissions"] == today


@pytest.mark.asyncio
async def test_a_typed_message_behind_sub_work_that_does_not_fit_still_starts() -> None:
    """Four running: an older approval wake in a delegated session waits for the
    reserve, while a newer message the user typed into another one fits."""
    wake, typed = _queued_row("wake"), _queued_row("typed")
    for row in (wake, typed):
        row.metadata.delegated_by_session_id = "parent"
        row.metadata.llm_auth_provider = "codex"
    waiting = {
        "wake": _pyd_message(metadata={held._WAKE_KEY: True}),
        "typed": _pyd_message(),
    }
    db = MagicMock()
    db.get_latest_user_message_in_session = AsyncMock(side_effect=waiting.get)
    dispatched = AsyncMock()

    with (
        _patch_queued_list([wake, typed]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=AsyncMock(return_value=True)),
        patch.object(
            turn_queue,
            "claim_queued_session",
            new=AsyncMock(
                side_effect=lambda _row, *, sub_work: "full" if sub_work else "admitted"
            ),
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=dispatched),
    ):
        assert await turn_queue.dispatch_next_for_user("u1") is True

    assert dispatched.await_args.kwargs["session_id"] == "typed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_envelope",
    [None, {"tree_id": "t1", "depth": 1}],
    ids=["unrecorded-wake", "tree-refusal"],
)
async def test_a_refusal_that_cannot_be_posted_still_frees_the_slot(
    stored_envelope: dict | None,
) -> None:
    """The session leaves ``running`` and the next queued one is tried."""
    head = _queued_row()
    head.metadata.delegated_by_session_id = "parent"
    head.metadata.llm_auth_provider = "codex"
    metadata: dict = {held._WAKE_KEY: True}
    if stored_envelope:
        metadata["envelope"] = stored_envelope
    db = MagicMock()
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(
        return_value=_pyd_message(metadata=metadata)
    )

    with (
        patch.object(
            turn_queue, "list_queued_sessions", new=AsyncMock(side_effect=[[head], []])
        ),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=AsyncMock(return_value=True)),
        patch.object(
            turn_queue, "claim_queued_session", new=AsyncMock(return_value="admitted")
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch.object(turn_queue, "resolve_session_permissions", return_value=None),
        patch.object(
            turn_queue,
            "append_and_save_message",
            new=AsyncMock(side_effect=RuntimeError("db blip")),
        ),
        patch(
            "backend.copilot.executor.utils.dispatch_turn",
            new=AsyncMock(side_effect=TreeRefusal("This tree has closed.")),
        ),
    ):
        assert await turn_queue.dispatch_next_for_user("u1") is False

    db.update_chat_session_status.assert_awaited_once_with(
        session_id="s1", expect_status="running", status="idle"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [RuntimeError("db blip"), asyncio.CancelledError()], ids=type
)
async def test_a_failure_between_the_claim_and_the_dispatch_requeues_the_turn(
    failure: BaseException,
) -> None:
    """Not left ``running`` with no turn to end it: the next tick retries."""
    head = _queued_row()
    head.metadata.llm_auth_provider = "codex"
    db = MagicMock()
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(side_effect=failure)

    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=AsyncMock(return_value=True)),
        patch.object(
            turn_queue, "claim_queued_session", new=AsyncMock(return_value="admitted")
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
    ):
        with pytest.raises(type(failure)):
            await turn_queue.dispatch_next_for_user("u1")

    db.update_chat_session_status.assert_awaited_once_with(
        session_id="s1", expect_status="running", status="queued"
    )


@pytest.mark.asyncio
async def test_dispatch_returns_false_when_queue_empty() -> None:
    with _patch_queued_list([]):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is False


@pytest.mark.asyncio
async def test_dispatch_leaves_queued_when_user_paywalled() -> None:
    """A queued head whose owner has lapsed to NO_TIER stays queued —
    no transition fires."""
    db = MagicMock()
    db.update_chat_session_status = AsyncMock()
    with (
        _patch_queued_list([_queued_row()]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=True),
        ),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is False
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_codex_dispatch_skips_platform_billing_gates() -> None:
    head = _queued_row()
    head.metadata.llm_auth_provider = "codex"
    head.metadata.llm_credential_id = "cred-1"
    pending = _pyd_message()
    db = MagicMock()
    db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    dispatch_turn_mock = AsyncMock()
    paywall_check = AsyncMock(side_effect=AssertionError("platform paywall checked"))
    global_limits = AsyncMock(side_effect=AssertionError("USD limits fetched"))
    rate_limit_check = AsyncMock(side_effect=AssertionError("USD limit checked"))

    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=paywall_check,
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=global_limits,
        ),
        patch(
            "backend.copilot.turn_queue.check_rate_limit",
            new=rate_limit_check,
        ),
        patch.object(
            turn_queue,
            "has_codex_access",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.executor.utils.dispatch_turn",
            new=dispatch_turn_mock,
        ),
        patch.object(
            turn_queue,
            "invalidate_session_cache",
            new=AsyncMock(),
        ),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")

    assert promoted is True
    paywall_check.assert_not_awaited()
    global_limits.assert_not_awaited()
    rate_limit_check.assert_not_awaited()
    dispatch_turn_mock.assert_awaited_once()
    assert dispatch_turn_mock.call_args.kwargs["llm_auth_provider"] == "codex"
    assert dispatch_turn_mock.call_args.kwargs["llm_credential_id"] == "cred-1"


@pytest.mark.asyncio
async def test_promotion_rechecks_the_advanced_tier_before_spending() -> None:
    """A queued Advanced turn is a decision to spend, taken later.

    Found by a blue-team pass: promotion already re-checked the paywall, the
    rate limit and Codex access, because a turn can wait long enough for any
    of them to change -- but not the tier. A user who queued Advanced turns
    and then downgraded had them dispatched anyway.
    """
    head = _queued_row()
    head.metadata.llm_auth_provider = "platform"
    pending = _pyd_message(metadata={"model": "advanced"})
    db = MagicMock()
    db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    dispatch_turn_mock = AsyncMock()
    entitled = AsyncMock(return_value=False)
    claim = AsyncMock(return_value="admitted")

    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(1, 1, None)),
        ),
        patch("backend.copilot.turn_queue.check_rate_limit", new=AsyncMock()),
        patch.object(turn_queue, "claim_queued_session", new=claim),
        patch.object(turn_queue, "advanced_tier_entitled", new=entitled),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=dispatch_turn_mock),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")

    assert promoted is False
    entitled.assert_awaited_once_with("u1")
    dispatch_turn_mock.assert_not_awaited()
    # Left queued rather than quietly re-run on Standard: nothing here
    # changes what a turn runs on without being asked.
    claim.assert_not_awaited()


@pytest.mark.asyncio
async def test_promotion_refuses_when_the_entitlement_cannot_be_resolved() -> None:
    """An outage is not permission to spend, and not a reason to lose the turn."""
    head = _queued_row()
    head.metadata.llm_auth_provider = "platform"
    pending = _pyd_message(metadata={"model": "advanced"})
    db = MagicMock()
    db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    dispatch_turn_mock = AsyncMock()
    claim = AsyncMock(return_value="admitted")

    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(1, 1, None)),
        ),
        patch("backend.copilot.turn_queue.check_rate_limit", new=AsyncMock()),
        patch.object(turn_queue, "claim_queued_session", new=claim),
        patch.object(
            turn_queue,
            "advanced_tier_entitled",
            new=AsyncMock(side_effect=turn_queue.EntitlementUnavailable("down")),
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=dispatch_turn_mock),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")

    assert promoted is False
    dispatch_turn_mock.assert_not_awaited()
    claim.assert_not_awaited()


@pytest.mark.parametrize(
    "queue, promoted",
    [
        # The user's own message goes first, though sub-work queued earlier.
        ([("sub", "parent", "platform"), ("chat", None, "platform")], "chat"),
        # One that may not start (no Codex access) does not hold up the rest.
        ([("chat", None, "codex"), ("sub", "parent", "platform")], "sub"),
        # With nothing of the user's queued, sub-work is promoted.
        ([("sub", "parent", "platform")], "sub"),
        # Within each group the oldest goes first.
        ([("older", None, "platform"), ("newer", None, "platform")], "older"),
        ([("older", "parent", "platform"), ("newer", "parent", "platform")], "older"),
    ],
)
@pytest.mark.asyncio
async def test_promotion_order(
    queue: list[tuple[str, str | None, str]], promoted: str
) -> None:
    rows = [_queued_row(session_id) for session_id, _, _ in queue]
    for row, (_, delegated_by, route) in zip(rows, queue):
        row.metadata.delegated_by_session_id = delegated_by
        row.metadata.llm_auth_provider = route
    # A delegated row's turn is sub-work: an approval wake held under its tree.
    sub_work = {
        held._WAKE_KEY: True,
        "envelope": TurnEnvelope(tree_id="tree", depth=1).model_dump(mode="json"),
    }
    waiting = {
        session_id: _pyd_message(
            metadata=sub_work if delegated_by else {"model": "standard"}
        )
        for session_id, delegated_by, _ in queue
    }
    db = MagicMock()
    db.get_latest_user_message_in_session = AsyncMock(side_effect=waiting.get)
    claim = AsyncMock(return_value="admitted")
    dispatched = AsyncMock()

    with (
        _patch_queued_list(rows),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=AsyncMock(return_value=False)),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(1, 1, None)),
        ),
        patch("backend.copilot.turn_queue.check_rate_limit", new=AsyncMock()),
        patch.object(turn_queue, "claim_queued_session", new=claim),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=dispatched),
    ):
        assert await turn_queue.dispatch_next_for_user("u1") is True

    assert claim.await_args.args[0].session_id == promoted
    assert dispatched.await_args.kwargs["session_id"] == promoted


@pytest.mark.asyncio
async def test_an_advanced_turn_without_the_tier_does_not_hold_up_the_queue() -> None:
    """Skipped like any session that may not start yet, and the per-user checks
    are made once however many sessions are tried."""
    chat, sub = _queued_row("chat"), _queued_row("sub")
    sub.metadata.delegated_by_session_id = "parent"
    models = {"chat": "advanced", "sub": "standard"}
    db = MagicMock()
    db.get_latest_user_message_in_session = AsyncMock(
        side_effect=lambda sid: _pyd_message(metadata={"model": models[sid]})
    )
    paywalled = AsyncMock(return_value=False)
    claim = AsyncMock(return_value="admitted")

    with (
        _patch_queued_list([chat, sub]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "is_user_paywalled", new=paywalled),
        patch.object(
            turn_queue,
            "get_global_rate_limits",
            new=AsyncMock(return_value=(1, 1, None)),
        ),
        patch.object(turn_queue, "check_rate_limit", new=AsyncMock()),
        patch.object(
            turn_queue, "advanced_tier_entitled", new=AsyncMock(return_value=False)
        ),
        patch.object(turn_queue, "claim_queued_session", new=claim),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=AsyncMock()),
    ):
        assert await turn_queue.dispatch_next_for_user("u1") is True

    assert claim.await_args.args[0].session_id == "sub"
    paywalled.assert_awaited_once_with("u1")


@pytest.mark.asyncio
async def test_a_degraded_rate_limit_service_leaves_the_whole_queue() -> None:
    chat, sub = _queued_row("chat"), _queued_row("sub")
    sub.metadata.delegated_by_session_id = "parent"
    sub.metadata.llm_auth_provider = "codex"
    codex = AsyncMock(return_value=True)
    db = MagicMock()
    db.get_latest_user_message_in_session = AsyncMock(return_value=_pyd_message())
    claim = AsyncMock(return_value="admitted")

    with (
        _patch_queued_list([chat, sub]),
        patch.object(turn_queue, "chat_db", return_value=db),
        # A dispatch that scanned on would start the turn: fail, not hang.
        patch("backend.copilot.executor.utils.dispatch_turn", new=AsyncMock()),
        patch.object(
            turn_queue, "is_user_paywalled", new=AsyncMock(return_value=False)
        ),
        patch.object(
            turn_queue,
            "get_global_rate_limits",
            new=AsyncMock(return_value=(1, 1, None)),
        ),
        patch.object(
            turn_queue,
            "check_rate_limit",
            new=AsyncMock(side_effect=turn_queue.RateLimitUnavailable()),
        ),
        patch.object(turn_queue, "has_codex_access", new=codex),
        patch.object(turn_queue, "claim_queued_session", new=claim),
    ):
        assert await turn_queue.dispatch_next_for_user("u1") is False

    codex.assert_not_awaited()
    claim.assert_not_awaited()


@pytest.mark.asyncio
async def test_promotion_does_not_recheck_the_tier_for_a_standard_turn() -> None:
    """The gate is on the paid tier only; Balanced promotes as before."""
    head = _queued_row()
    head.metadata.llm_auth_provider = "platform"
    pending = _pyd_message(metadata={"model": "standard"})
    db = MagicMock()
    db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    entitled = AsyncMock(side_effect=AssertionError("tier checked for Balanced"))

    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(1, 1, None)),
        ),
        patch("backend.copilot.turn_queue.check_rate_limit", new=AsyncMock()),
        patch.object(
            turn_queue, "claim_queued_session", new=AsyncMock(return_value="admitted")
        ),
        patch.object(turn_queue, "advanced_tier_entitled", new=entitled),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=AsyncMock()),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")

    assert promoted is True
    entitled.assert_not_awaited()


@pytest.mark.asyncio
async def test_promotion_uses_current_codex_route_not_stale_platform_tier() -> None:
    """Switching a queued turn to ChatGPT removes platform billing gates.

    The pending message keeps the tier selected when it was enqueued, but the
    session route is deliberately mutable so a blocked turn can continue on a
    subscription-backed connection.
    """
    head = _queued_row()
    head.metadata.llm_auth_provider = "codex"
    head.metadata.llm_credential_id = "cred-1"
    pending = _pyd_message(metadata={"model": "advanced"})
    db = MagicMock()
    db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    dispatch_turn_mock = AsyncMock()
    entitled = AsyncMock(side_effect=AssertionError("platform tier checked for Codex"))

    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=AsyncMock(return_value=True)),
        patch.object(
            turn_queue, "claim_queued_session", new=AsyncMock(return_value="admitted")
        ),
        patch.object(turn_queue, "advanced_tier_entitled", new=entitled),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=dispatch_turn_mock),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")

    assert promoted is True
    entitled.assert_not_awaited()
    assert dispatch_turn_mock.await_args.kwargs["llm_auth_provider"] == "codex"
    assert dispatch_turn_mock.await_args.kwargs["model"] == "advanced"


@pytest.mark.asyncio
async def test_microsoft_promotion_skips_platform_billing_gates() -> None:
    head = _queued_row()
    head.metadata.llm_auth_provider = "microsoft_365_copilot"
    head.metadata.llm_credential_id = "cred-microsoft"
    pending = _pyd_message(metadata={"model": "advanced"})
    db = MagicMock()
    db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    dispatch_turn_mock = AsyncMock()
    platform_gate = AsyncMock(
        side_effect=AssertionError("platform billing gate checked for Microsoft")
    )

    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "is_user_paywalled", new=platform_gate),
        patch.object(turn_queue, "advanced_tier_entitled", new=platform_gate),
        patch.object(
            turn_queue, "claim_queued_session", new=AsyncMock(return_value="admitted")
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
        patch("backend.copilot.executor.utils.dispatch_turn", new=dispatch_turn_mock),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")

    assert promoted is True
    platform_gate.assert_not_awaited()
    assert (
        dispatch_turn_mock.await_args.kwargs["llm_auth_provider"]
        == "microsoft_365_copilot"
    )


@pytest.mark.asyncio
async def test_codex_dispatch_leaves_queued_when_user_lacks_access() -> None:
    head = _queued_row()
    head.metadata.llm_auth_provider = "codex"
    head.metadata.llm_credential_id = "cred-1"
    db = MagicMock()
    db.update_chat_session_status = AsyncMock()
    dispatch_turn_mock = AsyncMock()
    access = AsyncMock(return_value=False)

    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch.object(turn_queue, "has_codex_access", new=access),
        patch(
            "backend.copilot.executor.utils.dispatch_turn",
            new=dispatch_turn_mock,
        ),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")

    assert promoted is False
    access.assert_awaited_once_with("u1")
    db.update_chat_session_status.assert_not_awaited()
    dispatch_turn_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_leaves_queued_on_rate_limit_exceeded() -> None:
    """Mid-queue rate-limit lapse: leave the head queued, the next tick
    re-validates."""
    from backend.copilot.rate_limit import RateLimitExceeded

    db = MagicMock()
    db.update_chat_session_status = AsyncMock()
    with (
        _patch_queued_list([_queued_row()]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(100, 1000, None)),
        ),
        patch(
            "backend.copilot.turn_queue.check_rate_limit",
            new=AsyncMock(
                side_effect=RateLimitExceeded(
                    "daily", resets_at=datetime.now(timezone.utc)
                )
            ),
        ),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is False
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_defers_on_rate_limit_unavailable() -> None:
    from backend.copilot.rate_limit import RateLimitUnavailable

    db = MagicMock()
    db.update_chat_session_status = AsyncMock()
    with (
        _patch_queued_list([_queued_row()]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(side_effect=RateLimitUnavailable()),
        ),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is False
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_happy_path_claims_and_dispatches() -> None:
    """All gates pass → claim session queued → running, build a TurnSlot,
    dispatch_turn, invalidate cache, return True."""
    head = _queued_row(session_id="s1")
    pending = _pyd_message(metadata={"mode": "extended_thinking"})
    db = MagicMock()
    db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    dispatch_turn_mock = AsyncMock()
    invalidate = AsyncMock()
    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(100, 1000, None)),
        ),
        patch(
            "backend.copilot.turn_queue.check_rate_limit",
            new=AsyncMock(),
        ),
        patch(
            "backend.copilot.executor.utils.dispatch_turn",
            new=dispatch_turn_mock,
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=invalidate),
    ):
        promoted = await turn_queue.dispatch_next_for_user("u1")
    assert promoted is True
    dispatch_turn_mock.assert_awaited_once()
    invalidate.assert_awaited_once_with("s1")
    # Claimed queued → running under the user's cap; nothing restored.
    assert db.admit_chat_session_turn.await_args.kwargs == {
        "session_id": "s1",
        "user_id": "u1",
        "expect_status": "queued",
        "capacity": turn_queue.get_running_turn_limit(),
    }
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_rolls_claim_back_on_dispatch_failure() -> None:
    """If ``dispatch_turn`` raises after claim, restore the session
    ``running`` → ``queued`` so the next tick can retry."""
    head = _queued_row(session_id="s1")
    pending = _pyd_message(metadata={"mode": "extended_thinking"})
    db = MagicMock()
    db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.get_latest_user_message_in_session = AsyncMock(return_value=pending)
    dispatch_turn_mock = AsyncMock(side_effect=RuntimeError("RabbitMQ blip"))
    with (
        _patch_queued_list([head]),
        patch.object(turn_queue, "chat_db", return_value=db),
        patch(
            "backend.copilot.turn_queue.is_user_paywalled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.get_global_rate_limits",
            new=AsyncMock(return_value=(100, 1000, None)),
        ),
        patch(
            "backend.copilot.turn_queue.check_rate_limit",
            new=AsyncMock(),
        ),
        patch(
            "backend.copilot.executor.utils.dispatch_turn",
            new=dispatch_turn_mock,
        ),
        patch.object(turn_queue, "invalidate_session_cache", new=AsyncMock()),
    ):
        with pytest.raises(RuntimeError, match="RabbitMQ blip"):
            await turn_queue.dispatch_next_for_user("u1")
    # Two transitions: claim then restore.  Redis-side meta cleanup is
    # ``dispatch_turn``'s responsibility (its own try/finally on the
    # ``committed`` flag), not the dispatcher's — see the
    # ``test_dispatch_turn_cleans_redis_on_enqueue_failure`` test in
    # ``executor/utils_test`` for that contract.
    db.admit_chat_session_turn.assert_awaited_once()
    db.update_chat_session_status.assert_awaited_once_with(
        session_id="s1", expect_status="running", status="queued"
    )
