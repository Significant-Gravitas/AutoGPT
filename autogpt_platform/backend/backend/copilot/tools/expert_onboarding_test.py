"""Tests for ExpertOnboardingTool."""

import json

import pytest

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.tools.expert_onboarding import (
    MAX_OPTION_LENGTH,
    MAX_OPTIONS,
    MAX_STEPS,
    MIN_STEPS,
    ExpertOnboardingTool,
)
from backend.copilot.tools.models import (
    ErrorResponse,
    ExpertOnboardingResponse,
    ResponseType,
)

EXPERT_ID = "1a5b1a10-6d10-4d7c-9d0d-2f6f1d9c0f11"


@pytest.fixture()
def tool() -> ExpertOnboardingTool:
    return ExpertOnboardingTool()


@pytest.fixture()
def session() -> ChatSession:
    return ChatSession.new(user_id="test-user", dry_run=False, expert_id=EXPERT_ID)


def steps(count: int = 2) -> list[dict]:
    return [
        {
            "question": f"Question {index}?",
            "options": ["Yes", "No"],
            "keyword": f"k{index}",
        }
        for index in range(count)
    ]


# ── Happy path ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_greeting_and_steps(
    tool: ExpertOnboardingTool, session: ChatSession
):
    result = await tool._execute(
        user_id="test-user",
        session=session,
        greeting="Hi, I'm Ada.",
        steps=[
            {
                "question": "Which services should I plug into?",
                "options": ["Linear", "GitHub", "Figma"],
                "keyword": "services",
            }
        ],
    )

    assert isinstance(result, ExpertOnboardingResponse)
    assert result.expert_id == EXPERT_ID
    assert result.session_id == session.session_id
    assert result.greeting == "Hi, I'm Ada."
    assert len(result.steps) == 1
    assert result.steps[0].keyword == "services"
    assert result.steps[0].options == ["Linear", "GitHub", "Figma"]
    assert result.message == "Which services should I plug into?"


@pytest.mark.asyncio
async def test_step_without_options_is_open_ended(
    tool: ExpertOnboardingTool, session: ChatSession
):
    result = await tool._execute(
        user_id="test-user",
        session=session,
        greeting="Hi.",
        steps=[{"question": "Anything else?", "keyword": "else"}],
    )

    assert isinstance(result, ExpertOnboardingResponse)
    assert result.steps[0].options == []


@pytest.mark.asyncio
async def test_missing_keyword_falls_back_to_index(
    tool: ExpertOnboardingTool, session: ChatSession
):
    result = await tool._execute(
        user_id="test-user",
        session=session,
        greeting="Hi.",
        steps=[{"question": "First?"}, {"question": "Second?"}],
    )

    assert isinstance(result, ExpertOnboardingResponse)
    assert [step.keyword for step in result.steps] == ["step-0", "step-1"]


# ── Caps and cleaning ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_steps_are_capped(tool: ExpertOnboardingTool, session: ChatSession):
    result = await tool._execute(
        user_id="test-user",
        session=session,
        greeting="Hi.",
        steps=steps(MAX_STEPS + 4),
    )

    assert isinstance(result, ExpertOnboardingResponse)
    assert len(result.steps) == MAX_STEPS


@pytest.mark.asyncio
async def test_options_are_capped_and_trimmed(
    tool: ExpertOnboardingTool, session: ChatSession
):
    result = await tool._execute(
        user_id="test-user",
        session=session,
        greeting="Hi.",
        steps=[
            {
                "question": "Pick one",
                "options": ["  Linear  "]
                + ["x" * (MAX_OPTION_LENGTH + 50)]
                + [f"opt{i}" for i in range(MAX_OPTIONS + 5)],
            }
        ],
    )

    assert isinstance(result, ExpertOnboardingResponse)
    options = result.steps[0].options
    assert len(options) == MAX_OPTIONS
    assert options[0] == "Linear"
    assert all(len(option) <= MAX_OPTION_LENGTH for option in options)


@pytest.mark.asyncio
async def test_repeated_options_and_keywords_collapse(
    tool: ExpertOnboardingTool, session: ChatSession
):
    result = await tool._execute(
        user_id="test-user",
        session=session,
        greeting="Hi.",
        steps=[
            {"question": "A?", "options": ["Yes", "Yes", "No"], "keyword": "dup"},
            {"question": "B?", "options": ["Yes"], "keyword": "dup"},
        ],
    )

    assert isinstance(result, ExpertOnboardingResponse)
    assert len(result.steps) == 1
    assert result.steps[0].options == ["Yes", "No"]


@pytest.mark.asyncio
async def test_non_string_options_are_dropped(
    tool: ExpertOnboardingTool, session: ChatSession
):
    result = await tool._execute(
        user_id="test-user",
        session=session,
        greeting="Hi.",
        steps=[{"question": "A?", "options": ["Yes", 7, None, ["No"]]}],
    )

    assert isinstance(result, ExpertOnboardingResponse)
    assert result.steps[0].options == ["Yes"]


# ── Refusals ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refuses_outside_an_expert_session(tool: ExpertOnboardingTool):
    plain = ChatSession.new(user_id="test-user", dry_run=False)

    result = await tool._execute(
        user_id="test-user", session=plain, greeting="Hi.", steps=steps()
    )

    assert isinstance(result, ErrorResponse)
    assert "expert" in result.message.lower()


def onboarding_call() -> ChatMessage:
    """The assistant row the runtime appends BEFORE the tool executes."""
    return ChatMessage(
        role="assistant",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "expert_onboarding", "arguments": "{}"},
            }
        ],
    )


def tool_result(payload: dict) -> ChatMessage:
    return ChatMessage(role="tool", content=json.dumps(payload), tool_call_id="call-1")


@pytest.mark.asyncio
async def test_refuses_a_second_card_in_the_same_chat(
    tool: ExpertOnboardingTool, session: ChatSession
):
    session.messages.append(onboarding_call())
    session.messages.append(
        tool_result({"type": ResponseType.EXPERT_ONBOARDING.value, "steps": []})
    )

    result = await tool._execute(
        user_id="test-user", session=session, greeting="Hi.", steps=steps()
    )

    assert isinstance(result, ErrorResponse)
    assert "already" in result.message.lower()


@pytest.mark.asyncio
async def test_a_failed_first_attempt_does_not_block_a_retry(
    tool: ExpertOnboardingTool, session: ChatSession
):
    """The tool-call row lands before the tool runs, so a rejected first call
    would otherwise convince the guard the hire was already onboarded."""
    session.messages.append(onboarding_call())
    session.messages.append(
        tool_result({"type": ResponseType.ERROR.value, "message": "bad arguments"})
    )

    result = await tool._execute(
        user_id="test-user", session=session, greeting="Hi.", steps=steps(3)
    )

    assert isinstance(result, ExpertOnboardingResponse)
    assert len(result.steps) == 3


@pytest.mark.asyncio
async def test_an_unparsable_tool_row_does_not_block(
    tool: ExpertOnboardingTool, session: ChatSession
):
    session.messages.append(ChatMessage(role="tool", content="not json"))

    result = await tool._execute(
        user_id="test-user", session=session, greeting="Hi.", steps=steps(3)
    )

    assert isinstance(result, ExpertOnboardingResponse)


def test_schema_declares_the_step_range(tool: ExpertOnboardingTool):
    steps_schema = tool.parameters["properties"]["steps"]

    assert steps_schema["minItems"] == MIN_STEPS
    assert steps_schema["maxItems"] == MAX_STEPS


@pytest.mark.asyncio
async def test_rejects_empty_greeting(tool: ExpertOnboardingTool, session: ChatSession):
    with pytest.raises(ValueError):
        await tool._execute(
            user_id="test-user", session=session, greeting="   ", steps=steps()
        )


@pytest.mark.asyncio
async def test_rejects_empty_steps(tool: ExpertOnboardingTool, session: ChatSession):
    with pytest.raises(ValueError):
        await tool._execute(
            user_id="test-user", session=session, greeting="Hi.", steps=[]
        )


@pytest.mark.asyncio
async def test_rejects_steps_with_no_valid_entry(
    tool: ExpertOnboardingTool, session: ChatSession
):
    with pytest.raises(ValueError):
        await tool._execute(
            user_id="test-user",
            session=session,
            greeting="Hi.",
            steps=["not a dict", {"question": "   "}],
        )
