"""Tests for AskQuestionTool."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import TaskStatus

from backend.copilot import turn_end
from backend.copilot.model import ChatSession, clear_pending_question
from backend.copilot.tools.ask_question import (
    MAX_DEFAULT_GUESS_LENGTH,
    MAX_OPTION_LENGTH,
    MAX_OPTIONS,
    MAX_QUESTIONS,
    AskQuestionTool,
)
from backend.copilot.tools.models import QUESTION_REASONS, ClarificationNeededResponse


@pytest.fixture(autouse=True)
def task_store() -> Iterator[MagicMock]:
    """Stand in for the task-status writes that asking and replying make."""
    store = MagicMock(move_session_tasks=AsyncMock(return_value=1))
    with (
        patch(
            "backend.copilot.tools.ask_question.task_db",
            MagicMock(return_value=store),
        ),
        patch("backend.copilot.model.task_db", MagicMock(return_value=store)),
    ):
        yield store


@pytest.fixture()
def tool() -> AskQuestionTool:
    return AskQuestionTool()


@pytest.fixture()
def session() -> ChatSession:
    return ChatSession.new(user_id="test-user", dry_run=False)


# ── Happy paths ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_question(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "Which channel?", "keyword": "channel"}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.message == "Which channel?"
    assert result.session_id == session.session_id
    assert len(result.questions) == 1
    assert result.questions[0].question == "Which channel?"
    assert result.questions[0].keyword == "channel"


@pytest.mark.asyncio
async def test_single_question_with_options(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {
                "question": "Which channel?",
                "options": ["Email", "Slack", "Google Docs"],
                "keyword": "channel",
            }
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    q = result.questions[0]
    assert q.example == "Email, Slack, Google Docs"
    assert q.options == ["Email", "Slack", "Google Docs"]


@pytest.mark.asyncio
async def test_allow_multiple_marks_the_question_multi_select(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {
                "question": "What areas should they own?",
                "options": ["Research", "Outreach", "Reporting"],
                "allow_multiple": True,
                "keyword": "areas",
            }
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].allow_multiple is True


@pytest.mark.asyncio
async def test_questions_are_single_select_by_default(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {
                "question": "Which channel?",
                "options": ["Email", "Slack"],
                "keyword": "channel",
            }
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].allow_multiple is False


@pytest.mark.asyncio
async def test_allow_multiple_accepts_the_string_a_model_may_send(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {
                "question": "Which areas?",
                "options": ["Research", "Outreach"],
                "allow_multiple": " TRUE ",
                "keyword": "areas",
            },
            {
                "question": "Which channel?",
                "options": ["Email", "Slack"],
                "allow_multiple": "no",
                "keyword": "channel",
            },
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert [q.allow_multiple for q in result.questions] == [True, False]


@pytest.mark.asyncio
async def test_allow_multiple_is_dropped_without_options(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {
                "question": "Anything else?",
                "allow_multiple": True,
                "keyword": "notes",
            }
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].allow_multiple is False


@pytest.mark.asyncio
async def test_allow_multiple_is_advertised_in_the_tool_schema(
    tool: AskQuestionTool,
):
    item = tool.parameters["properties"]["questions"]["items"]
    assert item["properties"]["allow_multiple"]["type"] == "boolean"
    assert "allow_multiple" not in item["required"]


@pytest.mark.asyncio
async def test_options_are_stripped_and_deduped(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {
                "question": "Which channel?",
                "options": ["  Email  ", "Slack", "   ", "Email"],
                "keyword": "channel",
            }
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    q = result.questions[0]
    assert q.options == ["Email", "Slack"]
    assert q.example == "Email, Slack"


@pytest.mark.asyncio
async def test_multiple_questions(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {
                "question": "Which channel?",
                "options": ["Email", "Slack"],
                "keyword": "channel",
            },
            {
                "question": "How often?",
                "options": ["Daily", "Weekly"],
                "keyword": "frequency",
            },
            {"question": "Any extra notes?"},
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 3
    assert result.message == "Which channel?; How often?; Any extra notes?"

    assert result.questions[0].keyword == "channel"
    assert result.questions[0].example == "Email, Slack"
    assert result.questions[1].keyword == "frequency"
    assert result.questions[2].keyword == "question-2"
    assert result.questions[2].example is None


# ── Keyword handling ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_keyword_gets_index_fallback(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "First?"}, {"question": "Second?"}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].keyword == "question-0"
    assert result.questions[1].keyword == "question-1"


@pytest.mark.asyncio
async def test_null_keyword_gets_index_fallback(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "First?", "keyword": None}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].keyword == "question-0"


@pytest.mark.asyncio
async def test_duplicate_keywords_preserved(
    tool: AskQuestionTool, session: ChatSession
):
    """Frontend normalizeClarifyingQuestions() handles dedup."""
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {"question": "First?", "keyword": "same"},
            {"question": "Second?", "keyword": "same"},
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].keyword == "same"
    assert result.questions[1].keyword == "same"


# ── Options filtering ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_options_preserves_falsy_strings(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "Pick", "options": ["0", "1", "2"]}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].example == "0, 1, 2"


@pytest.mark.asyncio
async def test_options_filters_none_and_empty(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "Pick", "options": ["Email", "", "Slack", None]}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].example == "Email, Slack"


@pytest.mark.asyncio
async def test_options_drops_non_strings_instead_of_coercing(
    tool: AskQuestionTool, session: ChatSession
):
    # str() would surface a Python repr like "['a']" as a tappable choice, and
    # the frontend's recovery path only keeps strings anyway.
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "Pick", "options": [None, 42, "Slack", ["a"]]}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].options == ["Slack"]


@pytest.mark.asyncio
async def test_options_are_capped_in_count_and_length(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {"question": "Pick", "options": [f"opt-{i}" for i in range(60)]},
            {"question": "Pick", "options": ["x" * 500]},
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions[0].options) == MAX_OPTIONS
    assert result.questions[0].options[0] == "opt-0"
    assert result.questions[1].options == ["x" * MAX_OPTION_LENGTH]


@pytest.mark.asyncio
async def test_options_stop_scanning_past_the_bound(
    tool: AskQuestionTool, session: ChatSession
):
    # Blanks never reach the cap, so only the scan bound stops the walk.
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "Pick", "options": [""] * 5000 + ["Slack"]}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].options == []


@pytest.mark.asyncio
async def test_questions_are_capped(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": f"Q{i}"} for i in range(40)],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == MAX_QUESTIONS


@pytest.mark.asyncio
async def test_no_options_gives_none_example(
    tool: AskQuestionTool, session: ChatSession
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "Thoughts?"}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].example is None


# ── Invalid input handling ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_skips_non_dict_items(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=["not-a-dict", {"question": "Valid?", "keyword": "v"}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 1
    assert result.questions[0].question == "Valid?"


@pytest.mark.asyncio
async def test_skips_empty_question_items(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {"keyword": "missing-question"},
            {"question": ""},
            {"question": "  Valid  ", "keyword": "v"},
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 1
    assert result.questions[0].question == "Valid"


@pytest.mark.asyncio
async def test_rejects_all_invalid_items(tool: AskQuestionTool, session: ChatSession):
    with pytest.raises(ValueError, match="at least one valid question"):
        await tool._execute(
            user_id=None,
            session=session,
            questions=[{"keyword": "no-q"}, "bad"],
        )


@pytest.mark.asyncio
async def test_rejects_empty_questions_array(
    tool: AskQuestionTool, session: ChatSession
):
    with pytest.raises(ValueError, match="non-empty"):
        await tool._execute(user_id=None, session=session, questions=[])


@pytest.mark.asyncio
async def test_rejects_missing_questions(tool: AskQuestionTool, session: ChatSession):
    with pytest.raises(ValueError, match="non-empty"):
        await tool._execute(user_id=None, session=session)


@pytest.mark.asyncio
async def test_rejects_non_list_questions(tool: AskQuestionTool, session: ChatSession):
    with pytest.raises(ValueError, match="non-empty"):
        await tool._execute(
            user_id=None,
            session=session,
            questions="not-a-list",
        )


# ── Home "Needs You" hand-off ────────────────────────────────────────


@pytest.mark.asyncio
async def test_asking_parks_the_question_on_the_session(
    tool: AskQuestionTool, session: ChatSession
):
    db = MagicMock()
    db.set_session_pending_question = AsyncMock()
    with patch(
        "backend.copilot.tools.ask_question.chat_db", MagicMock(return_value=db)
    ):
        await tool._execute(
            user_id=None,
            session=session,
            questions=[{"question": "Monday or Friday?"}],
        )

    assert session.metadata.pending_question is not None
    assert session.metadata.pending_question.text == "Monday or Friday?"
    db.set_session_pending_question.assert_awaited_once()
    assert db.set_session_pending_question.await_args.args[0] == session.session_id
    assert db.set_session_pending_question.await_args.args[1] == session.user_id


@pytest.mark.asyncio
async def test_a_failed_write_never_costs_the_user_the_question(
    tool: AskQuestionTool, session: ChatSession
):
    db = MagicMock()
    db.set_session_pending_question = AsyncMock(side_effect=RuntimeError("down"))
    with patch(
        "backend.copilot.tools.ask_question.chat_db", MagicMock(return_value=db)
    ):
        result = await tool._execute(
            user_id=None,
            session=session,
            questions=[{"question": "Monday or Friday?"}],
        )

    assert isinstance(result, ClarificationNeededResponse)


@pytest.mark.asyncio
async def test_replying_clears_the_pending_question(
    tool: AskQuestionTool, session: ChatSession
):
    db = MagicMock()
    db.set_session_pending_question = AsyncMock()
    db.clear_session_pending_question = AsyncMock()
    with (
        patch("backend.copilot.tools.ask_question.chat_db", MagicMock(return_value=db)),
        patch("backend.copilot.model.chat_db", MagicMock(return_value=db)),
    ):
        await tool._execute(
            user_id=None,
            session=session,
            questions=[{"question": "Monday or Friday?"}],
        )
        await clear_pending_question(session)

    assert session.metadata.pending_question is None
    db.clear_session_pending_question.assert_awaited_once_with(
        session.session_id, session.user_id
    )


@pytest.mark.asyncio
async def test_clearing_a_session_with_no_question_touches_nothing(
    session: ChatSession,
):
    db = MagicMock()
    db.clear_session_pending_question = AsyncMock()
    with patch("backend.copilot.model.chat_db", MagicMock(return_value=db)):
        await clear_pending_question(session)

    db.clear_session_pending_question.assert_not_called()


# ── Reason and default guess ─────────────────────────────────────────


@pytest.fixture()
def quiet_db():
    """Stand in for the pending-question write, so these tests need no DB."""
    with patch(
        "backend.copilot.tools.ask_question.chat_db",
        MagicMock(return_value=MagicMock(set_session_pending_question=AsyncMock())),
    ):
        yield


@pytest.mark.asyncio
async def test_reason_and_default_guess_reach_the_card(
    tool: AskQuestionTool, session: ChatSession, quiet_db
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {
                "question": "Which team gets the report?",
                "reason": " Goal_Ambiguous ",
                "default_guess": "  The sales team  ",
            }
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    question = result.questions[0]
    assert question.reason == "goal_ambiguous"
    assert question.default_guess == "The sales team"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason, default_guess",
    [(None, None), ("because", "   "), (7, ["x"])],
)
async def test_unknown_reason_and_blank_guess_read_as_missing(
    tool: AskQuestionTool, session: ChatSession, quiet_db, reason, default_guess
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {
                "question": "Which team?",
                "reason": reason,
                "default_guess": default_guess,
            }
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].reason is None
    assert result.questions[0].default_guess is None


@pytest.mark.asyncio
async def test_default_guess_is_capped(
    tool: AskQuestionTool, session: ChatSession, quiet_db
):
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "Which team?", "default_guess": "x" * 1000}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].default_guess == "x" * MAX_DEFAULT_GUESS_LENGTH


def test_schema_requires_a_reason_and_a_default_guess(tool: AskQuestionTool):
    item = tool.parameters["properties"]["questions"]["items"]

    assert item["required"] == ["question", "reason", "default_guess"]
    assert set(item["properties"]["reason"]["enum"]) == set(QUESTION_REASONS)


@pytest.mark.asyncio
async def test_a_valid_question_ends_the_turn(
    tool: AskQuestionTool, session: ChatSession, quiet_db
):
    try:
        await tool._execute(
            user_id=None, session=session, questions=[{"question": "Which team?"}]
        )
        assert turn_end.is_ending(session.session_id)
    finally:
        turn_end.clear(session.session_id)


@pytest.mark.asyncio
async def test_an_invalid_question_does_not_end_the_turn(
    tool: AskQuestionTool, session: ChatSession, quiet_db
):
    with pytest.raises(ValueError):
        await tool._execute(user_id=None, session=session, questions=[])
    assert not turn_end.is_ending(session.session_id)


@pytest.mark.asyncio
async def test_asking_parks_the_chats_tasks_and_replying_resumes_them(
    tool: AskQuestionTool, session: ChatSession, quiet_db, task_store: MagicMock
):
    db = MagicMock(clear_session_pending_question=AsyncMock())
    try:
        await tool._execute(
            user_id=None, session=session, questions=[{"question": "Which team?"}]
        )
        with patch("backend.copilot.model.chat_db", MagicMock(return_value=db)):
            await clear_pending_question(session)
    finally:
        turn_end.clear(session.session_id)

    assert [c.args for c in task_store.move_session_tasks.await_args_list] == [
        (
            session.session_id,
            session.user_id,
            [TaskStatus.TODO, TaskStatus.IN_PROGRESS],
            TaskStatus.NEEDS_INPUT,
        ),
        (
            session.session_id,
            session.user_id,
            [TaskStatus.NEEDS_INPUT],
            TaskStatus.IN_PROGRESS,
        ),
    ]
