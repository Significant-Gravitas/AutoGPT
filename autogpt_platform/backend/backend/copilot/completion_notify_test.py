"""Tests for rich copilot completion notification content."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import completion_notify
from backend.copilot.completion_notify import (
    AUTOPILOT_DISPLAY_NAME,
    build_completion_content,
    build_copilot_completion_payload,
    publish_copilot_completion,
)
from backend.copilot.model import ChatSessionInfo, ChatSessionMetadata, PendingQuestion
from backend.data import push_sender

_CN = "backend.copilot.completion_notify"


def _session(
    session_id: str = "sess-1",
    user_id: str = "user-1",
    title: str | None = "Q3 revenue report",
    expert_id: str | None = "expert-1",
    pending_question: str | None = None,
) -> ChatSessionInfo:
    metadata = ChatSessionMetadata()
    if pending_question is not None:
        metadata.pending_question = PendingQuestion(
            text=pending_question, asked_at=datetime.now(timezone.utc)
        )
    return ChatSessionInfo(
        session_id=session_id,
        user_id=user_id,
        title=title,
        usage=[],
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata=metadata,
        expert_id=expert_id,
    )


def _mock_dbs(session=None, expert=None):
    """Patch the module's db accessors; returns (chat_db, experts_db) mocks."""
    chat_db = MagicMock()
    chat_db.get_chat_session_metadata = AsyncMock(return_value=session)
    experts_db = MagicMock()
    experts_db.get_expert = AsyncMock(return_value=expert)
    return chat_db, experts_db


class TestBuildCompletionContent:
    def test_expert_completed_with_title(self):
        title, body = build_completion_content(
            status="completed",
            session_title="Q3 revenue report",
            expert_name="Maria",
            pending_question=None,
        )
        assert title == "Maria"
        assert body == "Finished: Q3 revenue report"

    def test_plain_session_uses_autopilot_title(self):
        title, body = build_completion_content(
            status="completed",
            session_title="Q3 revenue report",
            expert_name=None,
            pending_question=None,
        )
        assert title == AUTOPILOT_DISPLAY_NAME
        assert body == "Finished: Q3 revenue report"

    def test_completed_without_title_falls_back(self):
        title, body = build_completion_content(
            status="completed",
            session_title=None,
            expert_name=None,
            pending_question=None,
        )
        assert title == AUTOPILOT_DISPLAY_NAME
        assert body == "Task completed"

    def test_failed_with_title(self):
        title, body = build_completion_content(
            status="failed",
            session_title="Q3 revenue report",
            expert_name="Maria",
            pending_question=None,
        )
        assert title == "Maria"
        assert body == "Couldn't finish: Q3 revenue report"

    def test_failed_without_title_falls_back(self):
        title, body = build_completion_content(
            status="failed",
            session_title=None,
            expert_name=None,
            pending_question=None,
        )
        assert body == "Task failed"

    def test_pending_question_wins_over_title(self):
        title, body = build_completion_content(
            status="completed",
            session_title="Q3 revenue report",
            expert_name="Maria",
            pending_question="Which region should I focus on?",
        )
        assert title == "Maria"
        assert body == "Needs your answer: Which region should I focus on?"

    def test_pending_question_ignored_on_failure(self):
        _, body = build_completion_content(
            status="failed",
            session_title="Q3 revenue report",
            expert_name="Maria",
            pending_question="Which region?",
        )
        assert body == "Couldn't finish: Q3 revenue report"

    def test_long_values_are_truncated(self):
        title, body = build_completion_content(
            status="completed",
            session_title="x" * 500,
            expert_name="y" * 500,
            pending_question=None,
        )
        assert len(title) <= completion_notify.MAX_TITLE_LEN
        assert len(body) <= completion_notify.MAX_BODY_LEN
        assert title.endswith("…")
        assert body.endswith("…")

    def test_long_question_is_truncated(self):
        _, body = build_completion_content(
            status="completed",
            session_title=None,
            expert_name="Maria",
            pending_question="q" * 500,
        )
        assert len(body) <= completion_notify.MAX_BODY_LEN
        assert body.startswith("Needs your answer: ")

    def test_whitespace_is_collapsed(self):
        title, body = build_completion_content(
            status="completed",
            session_title="  Q3   revenue\nreport  ",
            expert_name="  Maria  ",
            pending_question=None,
        )
        assert title == "Maria"
        assert body == "Finished: Q3 revenue report"


class TestSafeAvatarUrl:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("/autogpt-characters/maria.webp", "/autogpt-characters/maria.webp"),
            (
                "https://cdn.example.com/avatars/m.png",
                "https://cdn.example.com/avatars/m.png",
            ),
            ("http://cdn.example.com/m.png", None),
            ("data:image/png;base64,abc", None),
            ("javascript:alert(1)", None),
            ("//evil.example.com/m.png", None),
            ("/\\evil.example.com/m.png", None),
            ("https://cdn.example.com/m.png\n", "https://cdn.example.com/m.png"),
        ],
    )
    def test_cases(self, raw, expected):
        assert completion_notify._safe_avatar_url(raw) == expected


class TestBuildPayload:
    async def test_full_enrichment_for_expert_session(self):
        expert = SimpleNamespace(name="Maria", avatar_url="/avatars/maria.webp")
        chat_db, experts_db = _mock_dbs(session=_session(), expert=expert)
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.type == "copilot_completion"
        assert payload.event == "session_completed"
        assert payload.session_id == "sess-1"
        assert payload.status == "completed"
        assert payload.title == "Maria"
        assert payload.body == "Finished: Q3 revenue report"
        assert payload.icon == "/avatars/maria.webp"
        assert payload.url == "/copilot?sessionId=sess-1"
        assert payload.expert_name == "Maria"
        assert payload.session_title == "Q3 revenue report"
        experts_db.get_expert.assert_awaited_once_with(
            "user-1", "expert-1", include_workflows=False
        )

    async def test_plain_session_skips_expert_lookup(self):
        chat_db, experts_db = _mock_dbs(session=_session(expert_id=None))
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.title == AUTOPILOT_DISPLAY_NAME
        assert payload.body == "Finished: Q3 revenue report"
        assert payload.icon is None
        assert payload.expert_name is None
        experts_db.get_expert.assert_not_awaited()

    async def test_missing_session_yields_generic_copy(self):
        chat_db, experts_db = _mock_dbs(session=None)
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.title == AUTOPILOT_DISPLAY_NAME
        assert payload.body == "Task completed"
        assert payload.url == "/copilot?sessionId=sess-1"

    async def test_session_lookup_error_yields_generic_copy(self):
        chat_db, experts_db = _mock_dbs()
        chat_db.get_chat_session_metadata = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.title == AUTOPILOT_DISPLAY_NAME
        assert payload.body == "Task completed"

    async def test_expert_lookup_error_keeps_session_body(self):
        chat_db, experts_db = _mock_dbs(session=_session())
        experts_db.get_expert = AsyncMock(side_effect=RuntimeError("db down"))
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.title == AUTOPILOT_DISPLAY_NAME
        assert payload.body == "Finished: Q3 revenue report"
        assert payload.icon is None

    async def test_archived_expert_falls_back_to_autopilot(self):
        chat_db, experts_db = _mock_dbs(session=_session(), expert=None)
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.title == AUTOPILOT_DISPLAY_NAME
        assert payload.body == "Finished: Q3 revenue report"

    async def test_owner_mismatch_skips_enrichment(self):
        chat_db, experts_db = _mock_dbs(session=_session(user_id="someone-else"))
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.title == AUTOPILOT_DISPLAY_NAME
        assert payload.body == "Task completed"
        assert payload.session_title is None
        experts_db.get_expert.assert_not_awaited()

    async def test_unsafe_avatar_url_is_dropped(self):
        expert = SimpleNamespace(name="Maria", avatar_url="javascript:alert(1)")
        chat_db, experts_db = _mock_dbs(session=_session(), expert=expert)
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.title == "Maria"
        assert payload.icon is None

    async def test_pending_question_produces_needs_body(self):
        chat_db, experts_db = _mock_dbs(
            session=_session(pending_question="Which region?"),
            expert=SimpleNamespace(name="Maria", avatar_url=None),
        )
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.body == "Needs your answer: Which region?"

    async def test_pending_question_ignored_for_failed_status(self):
        chat_db, experts_db = _mock_dbs(
            session=_session(pending_question="Which region?"),
            expert=SimpleNamespace(name="Maria", avatar_url=None),
        )
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "failed"
            )
        assert payload.body == "Couldn't finish: Q3 revenue report"

    async def test_oversized_session_title_stays_within_push_limit(self):
        """A user-set title has no length cap, but Web Push rejects payloads
        past ~4 KB — and repeated push failures get the subscription deleted.
        The structured fields are truncated so even a pathological title
        keeps the push payload small."""
        chat_db, experts_db = _mock_dbs(
            session=_session(title="t" * 5000),
            expert=SimpleNamespace(name="n" * 500, avatar_url=None),
        )
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
        ):
            payload = await build_copilot_completion_payload(
                "sess-1", "user-1", "completed"
            )
        assert payload.session_title is not None
        assert len(payload.session_title) <= completion_notify.MAX_BODY_LEN
        assert payload.expert_name is not None
        assert len(payload.expert_name) <= completion_notify.MAX_TITLE_LEN
        assert len(push_sender._build_push_payload(payload).encode()) < 4096


class TestPublishCopilotCompletion:
    async def test_publishes_enriched_event(self):
        chat_db, experts_db = _mock_dbs(
            session=_session(),
            expert=SimpleNamespace(name="Maria", avatar_url="/a.webp"),
        )
        bus = MagicMock()
        bus.publish = AsyncMock()
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
            patch(
                f"{_CN}.AsyncRedisNotificationEventBus",
                MagicMock(return_value=bus),
            ),
        ):
            await publish_copilot_completion("sess-1", "user-1", "completed")
        bus.publish.assert_awaited_once()
        event = bus.publish.await_args.args[0]
        assert event.user_id == "user-1"
        assert event.payload.title == "Maria"
        assert event.payload.session_id == "sess-1"

    async def test_builder_failure_still_publishes_bare_payload(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        with (
            patch(
                f"{_CN}.build_copilot_completion_payload",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch(
                f"{_CN}.AsyncRedisNotificationEventBus",
                MagicMock(return_value=bus),
            ),
        ):
            await publish_copilot_completion("sess-1", "user-1", "completed")
        bus.publish.assert_awaited_once()
        event = bus.publish.await_args.args[0]
        assert event.payload.session_id == "sess-1"
        assert event.payload.title is None

    async def test_publish_failure_is_swallowed(self):
        chat_db, experts_db = _mock_dbs(session=_session())
        bus = MagicMock()
        bus.publish = AsyncMock(side_effect=RuntimeError("redis down"))
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
            patch(
                f"{_CN}.AsyncRedisNotificationEventBus",
                MagicMock(return_value=bus),
            ),
        ):
            await publish_copilot_completion("sess-1", "user-1", "completed")

    @pytest.mark.parametrize("status", ["completed", "failed"])
    async def test_cancelled_stop_notifies_nothing(self, status):
        """A user-initiated Stop publishes no notification at all — and
        skips the enrichment reads, so no DB work happens either."""
        chat_db, experts_db = _mock_dbs(session=_session())
        bus = MagicMock()
        bus.publish = AsyncMock()
        with (
            patch(f"{_CN}.chat_db", MagicMock(return_value=chat_db)),
            patch(f"{_CN}.experts_db", MagicMock(return_value=experts_db)),
            patch(
                f"{_CN}.AsyncRedisNotificationEventBus",
                MagicMock(return_value=bus),
            ),
        ):
            await publish_copilot_completion("sess-1", "user-1", status, cancelled=True)
        bus.publish.assert_not_awaited()
        chat_db.get_chat_session_metadata.assert_not_awaited()
        experts_db.get_expert.assert_not_awaited()
