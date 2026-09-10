from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock

from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    STREAM_ERROR_MARKER,
)
from backend.copilot.markers import (
    append_error_marker,
    has_trailing_marker,
    is_error_marker,
    provider_failure_of,
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


class TestTheMarkerReachesTheDatabase:
    """Building the marker is not the same as saving it.

    The baseline yields the error and its consumer immediately breaks and
    closes the generator, so the tail of its ``finally`` -- including the
    session upsert -- never runs. A marker appended there is constructed
    correctly and then discarded, which is exactly what happened: the append
    reported success while the row never appeared in Postgres.

    The failure path therefore has to persist the marker itself, before it
    yields the error that ends the stream.

    These two are structural guards, not behavioural proof: they assert the
    ordering in the source rather than driving a real turn, because standing
    up that generator needs a provider, a session, tools and an execution
    context. The behavioural evidence is a live run against a deployment --
    a 404 leaves a non-retryable card and a dead endpoint leaves a retryable
    one -- and these keep the ordering from silently regressing afterwards.
    """

    def test_the_error_path_persists_before_it_yields(self) -> None:
        import inspect

        from backend.copilot.baseline import service

        source = inspect.getsource(service.stream_chat_completion_baseline)
        marker_at = source.find("append_error_marker(")
        assert marker_at != -1, "the failure path no longer records a marker"

        upsert_at = source.find("upsert_chat_session", marker_at)
        error_yield_at = source.find("yield StreamError(", marker_at)
        assert upsert_at != -1, "the marker is appended but never persisted"
        assert error_yield_at != -1

        # Persisting after the error is yielded is the bug this guards:
        # the consumer closes the generator on that yield.
        assert upsert_at < error_yield_at, (
            "the marker must be persisted before StreamError is yielded -- "
            "the consumer closes the generator on it, so anything after is "
            "not guaranteed to run"
        )

    def test_the_marker_is_not_left_to_the_finally_block(self) -> None:
        import inspect

        from backend.copilot.baseline import service

        source = inspect.getsource(service.stream_chat_completion_baseline)
        finally_at = source.rfind("\n    finally:")
        marker_at = source.find("append_error_marker(")
        assert marker_at != -1
        assert finally_at != -1
        assert marker_at < finally_at, (
            "the marker moved into finally, where generator teardown cuts it "
            "short at the first await"
        )


class TestTheMarkerCarriesTheFailure:
    """The prefix says retry-or-not. The envelope says what would fix it.

    A chat reopened tomorrow has only the row: without the failure on it,
    the most a card can offer is Try Again, which is the wrong advice for
    an expired login, a spent quota or a plan that excludes the connection.
    """

    def test_the_failure_rides_on_the_row(self) -> None:
        session = _session()
        append_error_marker(
            session,
            "You've hit this connection's limit",
            retryable=False,
            failure={"kind": "usage_limit", "authProvider": "codex", "resetsAt": None},
        )

        recorded = provider_failure_of(session.messages[-1])
        assert recorded is not None
        assert recorded["kind"] == "usage_limit"
        assert recorded["authProvider"] == "codex"

    def test_a_marker_without_one_reads_as_none(self) -> None:
        # Every marker written before this existed, and every failure the
        # classifier declined to name. Callers fall back to the prefix.
        session = _session()
        append_error_marker(session, "boom", retryable=True)
        assert provider_failure_of(session.messages[-1]) is None

    def test_an_ordinary_reply_is_never_read_as_a_failure(self) -> None:
        assert (
            provider_failure_of(ChatMessage(role="assistant", content="Here you go."))
            is None
        )

    def test_a_non_dict_payload_is_refused(self) -> None:
        # The bag is shared; a collision on the key must not crash a render.
        msg = ChatMessage(
            role="assistant",
            content=f"{COPILOT_ERROR_PREFIX} boom",
            metadata={"provider_failure": "not-a-dict"},
        )
        assert provider_failure_of(msg) is None


class TestTheFailureReachesTheDatabase:
    """Setting metadata on the row is not the same as saving it.

    ``_save_session_to_db`` builds each row field by field and
    ``add_chat_messages_batch`` maps them one by one, so a field is only
    persisted if it is named in both. ``metadata`` was named in neither --
    the single-message path persisted it, the batch path silently dropped it.
    """

    @pytest.mark.asyncio
    async def test_a_marker_reaches_the_database_layer_with_its_failure(
        self, mocker: pytest_mock.MockerFixture
    ) -> None:
        from backend.copilot import model as model_module

        captured: dict[str, object] = {}

        async def _capture(session_id: str, messages: list, start_sequence: int) -> int:
            captured["messages"] = messages
            return start_sequence

        fake_db = MagicMock()
        fake_db.add_chat_messages_batch = AsyncMock(side_effect=_capture)
        fake_db.get_chat_session_metadata = AsyncMock(return_value=None)
        fake_db.create_chat_session = AsyncMock(return_value=None)
        fake_db.update_chat_session = AsyncMock(return_value=None)
        mocker.patch.object(model_module, "chat_db", return_value=fake_db)

        session = _session()
        append_error_marker(
            session, "boom", retryable=False, failure={"kind": "auth_expired"}
        )
        await model_module._save_session_to_db(
            session, existing_message_count=0, skip_existence_check=True
        )

        rows = captured.get("messages")
        assert rows, "no rows reached the database layer"
        assert rows[0]["metadata"] == {"provider_failure": {"kind": "auth_expired"}}
