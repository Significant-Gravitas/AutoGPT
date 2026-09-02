"""Tests for AskQuestionTool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession, clear_pending_question
from backend.copilot.tools.ask_question import (
    MAX_OPTION_LENGTH,
    MAX_OPTIONS,
    MAX_QUESTIONS,
    AskQuestionTool,
)
from backend.copilot.tools.models import ClarificationNeededResponse


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
