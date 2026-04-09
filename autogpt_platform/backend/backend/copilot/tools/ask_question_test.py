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


# ── Single-question (backward-compatible) ────────────────────────────


@pytest.mark.asyncio
async def test_execute_with_options(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        question="Which channel?",
        options=["Email", "Slack", "Google Docs"],
        keyword="channel",
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.message == "Which channel?"
    assert result.session_id == session.session_id
    assert len(result.questions) == 1

    q = result.questions[0]
    assert q.question == "Which channel?"
    assert q.keyword == "channel"
    assert q.example == "Email, Slack, Google Docs"


@pytest.mark.asyncio
async def test_execute_without_options(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        question="What format do you want?",
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.message == "What format do you want?"
    assert len(result.questions) == 1

    q = result.questions[0]
    assert q.question == "What format do you want?"
    assert q.keyword == ""
    assert q.example is None


@pytest.mark.asyncio
async def test_execute_with_keyword_only(tool: AskQuestionTool, session: ChatSession):
    result = await tool._execute(
        user_id=None,
        session=session,
        question="How often should it run?",
        keyword="trigger",
    )

    assert isinstance(result, ClarificationNeededResponse)
    q = result.questions[0]
    assert q.keyword == "trigger"
    assert q.example is None


@pytest.mark.asyncio
async def test_execute_rejects_empty_question(
    tool: AskQuestionTool, session: ChatSession
):
    with pytest.raises(ValueError, match="non-empty"):
        await tool._execute(user_id=None, session=session, question="")

    with pytest.raises(ValueError, match="non-empty"):
        await tool._execute(user_id=None, session=session, question="   ")


@pytest.mark.asyncio
async def test_execute_coerces_invalid_options(
    tool: AskQuestionTool, session: ChatSession
):
    """LLM may send options as a string instead of a list; should not crash."""
    result = await tool._execute(
        user_id=None,
        session=session,
        question="Pick one",
        options="not-a-list",  # type: ignore[arg-type]
    )

    assert isinstance(result, ClarificationNeededResponse)
    q = result.questions[0]
    assert q.example is None


# ── Multi-question ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_multiple_questions(tool: AskQuestionTool, session: ChatSession):
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
            {
                "question": "Any extra notes?",
            },
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 3
    assert result.message == "Which channel?; How often?; Any extra notes?"

    q0 = result.questions[0]
    assert q0.question == "Which channel?"
    assert q0.keyword == "channel"
    assert q0.example == "Email, Slack"

    q1 = result.questions[1]
    assert q1.question == "How often?"
    assert q1.keyword == "frequency"
    assert q1.example == "Daily, Weekly"

    q2 = result.questions[2]
    assert q2.question == "Any extra notes?"
    assert q2.keyword == "question-2"
    assert q2.example is None


@pytest.mark.asyncio
async def test_execute_multiple_questions_skips_invalid_items(
    tool: AskQuestionTool, session: ChatSession
):
    """Non-dict items and items without a question are skipped with a warning."""
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            "not-a-dict",
            {"keyword": "missing-question"},
            {"question": ""},
            {"question": "  Valid question  ", "keyword": "valid"},
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 1
    assert result.questions[0].question == "Valid question"
    assert result.questions[0].keyword == "valid"


@pytest.mark.asyncio
async def test_execute_multiple_questions_rejects_all_invalid(
    tool: AskQuestionTool, session: ChatSession
):
    """If every item in questions is invalid, raise ValueError."""
    with pytest.raises(ValueError, match="at least one valid question"):
        await tool._execute(
            user_id=None,
            session=session,
            questions=[{"keyword": "no-question"}, "bad"],
        )


@pytest.mark.asyncio
async def test_execute_multiple_questions_ignores_single_params(
    tool: AskQuestionTool, session: ChatSession
):
    """When 'questions' is provided, 'question'/'options'/'keyword' are ignored."""
    result = await tool._execute(
        user_id=None,
        session=session,
        question="Should be ignored",
        options=["A", "B"],
        keyword="ignored",
        questions=[
            {"question": "Real question?", "keyword": "real"},
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 1
    assert result.questions[0].question == "Real question?"
    assert result.questions[0].keyword == "real"


@pytest.mark.asyncio
async def test_execute_empty_questions_falls_back_to_single(
    tool: AskQuestionTool, session: ChatSession
):
    """An empty 'questions' list falls back to the single-question path."""
    result = await tool._execute(
        user_id=None,
        session=session,
        question="Fallback question?",
        questions=[],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 1
    assert result.questions[0].question == "Fallback question?"


# ── Edge cases (from review) ────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_with_none_session(tool: AskQuestionTool):
    """session_id should be None when session has no session_id."""

    class FakeSession:
        session_id = None

    result = await tool._execute(
        user_id=None,
        session=FakeSession(),  # type: ignore[arg-type]
        question="Does this work?",
    )
    assert isinstance(result, ClarificationNeededResponse)
    assert result.session_id is None
    assert result.questions[0].question == "Does this work?"


@pytest.mark.asyncio
async def test_execute_single_item_questions_array(
    tool: AskQuestionTool, session: ChatSession
):
    """A questions array with a single item should work like the single path."""
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[{"question": "Only one?", "keyword": "solo"}],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 1
    assert result.questions[0].question == "Only one?"
    assert result.questions[0].keyword == "solo"
    assert result.message == "Only one?"


@pytest.mark.asyncio
async def test_execute_duplicate_keywords_preserved(
    tool: AskQuestionTool, session: ChatSession
):
    """Duplicate keywords from the LLM are passed through; the frontend
    normalizes them via normalizeClarifyingQuestions()."""
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {"question": "First?", "keyword": "same"},
            {"question": "Second?", "keyword": "same"},
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 2
    assert result.questions[0].keyword == "same"
    assert result.questions[1].keyword == "same"


@pytest.mark.asyncio
async def test_execute_options_with_falsy_values(
    tool: AskQuestionTool, session: ChatSession
):
    """Falsy but valid option values like '0' should be preserved."""
    result = await tool._execute(
        user_id=None,
        session=session,
        question="Pick a number",
        options=["0", "1", "2"],
        keyword="number",
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].example == "0, 1, 2"


@pytest.mark.asyncio
async def test_execute_questions_as_non_list_falls_back(
    tool: AskQuestionTool, session: ChatSession
):
    """If 'questions' is a truthy non-list (e.g. a dict), fall back to single path."""
    result = await tool._execute(
        user_id=None,
        session=session,
        question="Fallback?",
        questions={"question": "Not a list"},  # type: ignore[arg-type]
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 1
    assert result.questions[0].question == "Fallback?"


@pytest.mark.asyncio
async def test_execute_multi_keyword_null_gets_fallback(
    tool: AskQuestionTool, session: ChatSession
):
    """If keyword is None (JSON null), use the question-{idx} fallback."""
    result = await tool._execute(
        user_id=None,
        session=session,
        questions=[
            {"question": "First?", "keyword": None},
            {"question": "Second?"},
        ],
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert len(result.questions) == 2
    assert result.questions[0].keyword == "question-0"
    assert result.questions[1].keyword == "question-1"


@pytest.mark.asyncio
async def test_execute_options_filters_empty_strings(
    tool: AskQuestionTool, session: ChatSession
):
    """Empty-string options should be filtered out to avoid artifacts."""
    result = await tool._execute(
        user_id=None,
        session=session,
        question="Pick one",
        options=["Email", "", "Slack", None],  # type: ignore[list-item]
        keyword="channel",
    )

    assert isinstance(result, ClarificationNeededResponse)
    assert result.questions[0].example == "Email, Slack"
