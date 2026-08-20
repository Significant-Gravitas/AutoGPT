from datetime import UTC, datetime

import pytest

from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    STREAM_ERROR_MARKER,
)
from backend.copilot.markers import (
    append_error_marker,
    has_trailing_marker,
    is_error_marker,
)
from backend.copilot.model import ChatMessage, ChatSession


def _session(messages: list[ChatMessage] | None = None) -> ChatSession:
    now = datetime(2026, 8, 20, tzinfo=UTC)
    return ChatSession(
        session_id="s1",
        user_id="u1",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=messages or [],
    )


class TestAFailureSurvivesTheStream:
    def test_a_failed_turn_leaves_something_behind(self) -> None:
        # Without this the chat shows the question and nothing after it.
        session = _session([ChatMessage(role="user", content="hi")])

        assert append_error_marker(session, "boom", retryable=True) is True
        assert len(session.messages) == 2
        assert session.messages[-1].role == "assistant"
        assert "boom" in (session.messages[-1].content or "")

    def test_the_provider_s_own_words_are_kept(self) -> None:
        session = _session()
        append_error_marker(
            session, "Can't reach the local LLM backend at :8099.", retryable=True
        )
        assert "8099" in (session.messages[-1].content or "")


class TestRetryIsOnlyOfferedWhenItCanWork:
    def test_a_transient_failure_offers_try_again(self) -> None:
        session = _session()
        append_error_marker(session, "hiccup", retryable=True)
        assert (session.messages[-1].content or "").startswith(
            COPILOT_RETRYABLE_ERROR_PREFIX
        )

    def test_a_failure_retrying_cannot_fix_does_not(self) -> None:
        # An expired login or a spent quota costs the user a retry to learn
        # the retry was never going to work.
        session = _session()
        append_error_marker(session, "Your plan does not include this", retryable=False)
        content = session.messages[-1].content or ""
        assert content.startswith(COPILOT_ERROR_PREFIX)
        assert not content.startswith(COPILOT_RETRYABLE_ERROR_PREFIX)


class TestOneFailureIsOneCard:
    @pytest.mark.parametrize(
        "existing",
        [
            f"{COPILOT_ERROR_PREFIX} already failed",
            f"{COPILOT_RETRYABLE_ERROR_PREFIX} already failed",
            STREAM_ERROR_MARKER,
        ],
    )
    def test_a_second_guard_does_not_stack_another_card(self, existing: str) -> None:
        # Several guards can fire on one failed turn.
        session = _session([ChatMessage(role="assistant", content=existing)])

        assert append_error_marker(session, "boom", retryable=True) is False
        assert len(session.messages) == 1

    def test_a_normal_reply_is_not_mistaken_for_a_marker(self) -> None:
        session = _session([ChatMessage(role="assistant", content="Here you go.")])

        assert has_trailing_marker(session) is False
        assert append_error_marker(session, "boom", retryable=True) is True

    def test_a_user_row_never_blocks_the_marker(self) -> None:
        session = _session([ChatMessage(role="user", content="hi")])
        assert has_trailing_marker(session) is False


class TestRecognisingMarkers:
    def test_both_error_prefixes_count(self) -> None:
        for prefix in (COPILOT_ERROR_PREFIX, COPILOT_RETRYABLE_ERROR_PREFIX):
            assert is_error_marker(ChatMessage(role="assistant", content=f"{prefix} x"))

    def test_ordinary_content_does_not(self) -> None:
        assert not is_error_marker(ChatMessage(role="assistant", content="hello"))

    def test_a_user_row_carrying_the_text_does_not(self) -> None:
        # Only the assistant writes markers; a user pasting the string in
        # must not make their own message render as an error card.
        assert not is_error_marker(
            ChatMessage(role="user", content=f"{COPILOT_ERROR_PREFIX} spoof")
        )

    def test_an_empty_session_has_no_trailing_marker(self) -> None:
        assert has_trailing_marker(_session()) is False
        assert has_trailing_marker(None) is False
