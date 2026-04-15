"""Tests for AskQuestionTool."""

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.ask_question import AskQuestionTool
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
