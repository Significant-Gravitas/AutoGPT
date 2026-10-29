from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock

from backend.copilot.config import CopilotLlmAuthProvider
from backend.copilot.model import ChatMessage, ChatSession, ChatSessionMetadata
from backend.copilot.segments import (
    Segment,
    segment_boundaries,
    segment_of,
    session_segment,
    stamp_segment,
)


def _session(
    auth_provider: CopilotLlmAuthProvider = "platform",
    credential_id: str | None = None,
    messages: list[ChatMessage] | None = None,
) -> ChatSession:
    now = datetime(2026, 8, 20, tzinfo=UTC)
    return ChatSession(
        session_id="s1",
        user_id="u1",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=messages or [],
        metadata=ChatSessionMetadata(
            llm_auth_provider=auth_provider,
            llm_credential_id=credential_id,
        ),
    )


def _assistant(
    auth_provider: CopilotLlmAuthProvider | None = None,
    credential_id: str | None = None,
    sequence: int | None = None,
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content="hi",
        llm_auth_provider=auth_provider,
        llm_credential_id=credential_id,
        sequence=sequence,
    )


class TestSegmentZero:
    def test_a_turn_that_recorded_nothing_reads_as_the_session_route(self) -> None:
        # Every row written before these columns existed. A backfill would
        # have had to invent what it could not know.
        session = _session("codex", "cred-1")
        segment = segment_of(_assistant(), session)
        assert segment == Segment("codex", "cred-1", is_segment_zero=True)
        assert segment.is_segment_zero is True

    def test_a_turn_that_recorded_its_route_is_believed(self) -> None:
        session = _session("platform", None)
        segment = segment_of(_assistant("codex", "cred-9"), session)
        assert segment == Segment("codex", "cred-9", is_segment_zero=False)
        assert segment.is_segment_zero is False

    def test_a_credential_without_a_provider_is_not_guessed_at(self) -> None:
        # It names an account without saying whose. Falling back is honest;
        # pairing it with the session's provider would fabricate a route.
        session = _session("platform", None)
        segment = segment_of(_assistant(None, "cred-9"), session)
        assert segment == Segment("platform", None, is_segment_zero=True)


class TestHistoryIsNotRewritten:
    def test_a_route_change_leaves_earlier_turns_alone(self) -> None:
        earlier = _assistant("codex", "cred-1")
        session = _session("platform", None, [earlier])

        # The session now points somewhere else entirely.
        assert segment_of(earlier, session) == Segment(
            "codex", "cred-1", is_segment_zero=False
        )

    def test_stamping_never_overwrites_a_turn_that_already_has_a_route(self) -> None:
        already = _assistant("codex", "cred-1")
        stamp_segment([already], 0, Segment("platform", None, is_segment_zero=True))
        assert already.llm_auth_provider == "codex"
        assert already.llm_credential_id == "cred-1"

    def test_only_this_run_s_turns_are_stamped(self) -> None:
        history = _assistant()
        fresh = _assistant()
        stamp_segment(
            [history, fresh], 1, Segment("codex", "cred-1", is_segment_zero=False)
        )
        assert history.llm_auth_provider is None
        assert fresh.llm_auth_provider == "codex"


class TestStamping:
    def test_a_user_row_is_never_given_a_route(self) -> None:
        user = ChatMessage(role="user", content="hi")
        stamp_segment([user], 0, Segment("codex", "cred-1", is_segment_zero=False))
        assert user.llm_auth_provider is None

    def test_a_row_already_flushed_is_flagged_for_backfill(self) -> None:
        # Mid-turn flush assigned a sequence before end-of-turn stamping.
        flushed = _assistant(sequence=4)
        stamp_segment([flushed], 0, Segment("codex", "cred-1", is_segment_zero=False))
        assert flushed.stamps_pending_save is True

    def test_an_unflushed_row_needs_no_backfill(self) -> None:
        fresh = _assistant()
        stamp_segment([fresh], 0, Segment("codex", "cred-1", is_segment_zero=False))
        assert fresh.stamps_pending_save is False


class TestBoundaries:
    def test_a_chat_that_never_moved_has_none(self) -> None:
        session = _session("platform", None)
        messages = [_assistant("platform"), _assistant("platform")]
        assert segment_boundaries(messages, session) == []

    def test_the_turn_the_route_changed_on_is_a_boundary(self) -> None:
        session = _session("platform", None)
        messages = [
            _assistant("platform"),
            _assistant("codex", "cred-1"),
            _assistant("codex", "cred-1"),
        ]
        assert segment_boundaries(messages, session) == [1]

    def test_moving_back_is_a_boundary_too(self) -> None:
        session = _session("platform", None)
        messages = [
            _assistant("codex", "cred-1"),
            _assistant("platform"),
        ]
        assert segment_boundaries(messages, session) == [0, 1]

    def test_switching_between_two_accounts_of_one_provider_counts(self) -> None:
        # Same provider, different subscription -- a real change of who pays.
        session = _session("codex", "cred-1")
        messages = [_assistant("codex", "cred-1"), _assistant("codex", "cred-2")]
        assert segment_boundaries(messages, session) == [1]

    def test_unstamped_rows_do_not_read_as_a_change(self) -> None:
        # User and tool rows carry no route, and neither do turns written
        # before the columns existed. Treating either as "back to segment
        # zero" would invent boundaries that never happened.
        session = _session("codex", "cred-1")
        messages = [
            _assistant("codex", "cred-1"),
            ChatMessage(role="user", content="next"),
            ChatMessage(role="tool", content="{}", tool_call_id="t1"),
            _assistant("codex", "cred-1"),
        ]
        assert segment_boundaries(messages, session) == []


class TestSessionSegment:
    def test_it_reports_where_the_chat_started(self) -> None:
        segment = session_segment(_session("codex", "cred-1"))
        assert segment.auth_provider == "codex"
        assert segment.credential_id == "cred-1"
        assert segment.is_segment_zero is True


@pytest.mark.parametrize(
    "a, b, equal",
    [
        (
            Segment("platform", None, is_segment_zero=True),
            Segment("platform", None, is_segment_zero=False),
            True,
        ),
        (
            Segment("codex", "c1", is_segment_zero=False),
            Segment("codex", "c2", is_segment_zero=False),
            False,
        ),
        (
            Segment("platform", None, is_segment_zero=False),
            Segment("codex", None, is_segment_zero=False),
            False,
        ),
    ],
)
def test_two_segments_match_on_the_route_not_on_where_it_was_read(
    a: Segment, b: Segment, equal: bool
) -> None:
    # Whether the answer came from the turn or the session is provenance,
    # not identity -- otherwise the first stamped turn of an unchanged chat
    # would read as a boundary.
    assert (a == b) is equal


class TestTheRouteSurvivesTheSavePath:
    """The stamp is worthless if it is dropped on the way to the database.

    ``_save_session_to_db`` builds each row's dict field by field rather than
    dumping the model, so a new field on ``ChatMessage`` reaches Postgres only
    if it is named there too. Stamping worked and the column existed, and the
    route still landed as NULL because that one list had not been updated.
    """

    @pytest.mark.asyncio
    async def test_a_stamped_turn_reaches_the_database_layer_with_its_route(
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

        session = _session("codex", "cred-1", [_assistant("codex", "cred-1")])
        await model_module._save_session_to_db(
            session, existing_message_count=0, skip_existence_check=True
        )

        rows = captured.get("messages")
        assert rows, "no rows reached the database layer"
        assistant_rows = [r for r in rows if r["role"] == "assistant"]
        assert assistant_rows, "the assistant row never reached the database layer"
        assert assistant_rows[0]["llm_auth_provider"] == "codex"
        assert assistant_rows[0]["llm_credential_id"] == "cred-1"
