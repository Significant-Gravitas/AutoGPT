"""Tests for ExpertOnboardingTool."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.expert_kickoff import (
    expert_kickoff_metadata,
    is_expert_kickoff_turn,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.tools import (
    execute_tool,
    get_available_tools,
    get_tool,
    kickoff_turn_disabled_tools,
)
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


# ── Kickoff-turn dispatch gate ───────────────────────────────────────
#
# The kickoff prompt tells the model to open the card and nothing else.
# These cover that sentence as an enforcement boundary: what a model can
# dispatch before the card exists, and after the user has answered it.


def _kickoff_session() -> ChatSession:
    session = ChatSession.new(user_id="alice", dry_run=False, expert_id=EXPERT_ID)
    session.messages.append(
        ChatMessage(
            role="user",
            content="You were just hired.",
            metadata=expert_kickoff_metadata(EXPERT_ID),
        )
    )
    return session


def test_the_kickoff_turn_offers_the_card_and_nothing_else():
    gated = kickoff_turn_disabled_tools()

    names = {
        schema["function"]["name"]
        for schema in get_available_tools(disabled_tools=gated)
    }

    assert names == {"expert_onboarding"}
    assert {"run_agent", "schedule_followup"} <= gated


@pytest.mark.asyncio
async def test_run_agent_is_refused_on_the_kickoff_turn():
    run_agent = get_tool("run_agent")
    assert run_agent is not None

    with patch.object(
        run_agent, "execute", new=AsyncMock(return_value="should never run")
    ) as execute_mock:
        result = await execute_tool(
            tool_name="run_agent",
            parameters={},
            user_id="alice",
            session=_kickoff_session(),
            tool_call_id="call-1",
            disabled_groups=(),
            disabled_tools=kickoff_turn_disabled_tools(),
        )

    execute_mock.assert_not_awaited()
    assert result.success is False
    assert ErrorResponse.model_validate_json(result.output).error == "tool_disabled"


@pytest.mark.asyncio
async def test_the_onboarding_card_itself_still_dispatches_on_the_kickoff_turn():
    onboarding = get_tool("expert_onboarding")
    assert onboarding is not None

    with patch.object(
        onboarding, "execute", new=AsyncMock(return_value="the card")
    ) as execute_mock:
        result = await execute_tool(
            tool_name="expert_onboarding",
            parameters={},
            user_id="alice",
            session=_kickoff_session(),
            tool_call_id="call-1",
            disabled_groups=(),
            disabled_tools=kickoff_turn_disabled_tools(),
        )

    execute_mock.assert_awaited_once()
    assert result == "the card"


@pytest.mark.asyncio
async def test_run_agent_dispatches_once_the_user_has_answered_the_card():
    """The gate is the kickoff turn, not onboarding state.

    A user who skipped the card, or one whose expert never managed to open
    it, still gets the work they ask for on their own turns.
    """
    session = _kickoff_session()
    session.messages.append(ChatMessage(role="user", content="Run the digest now."))
    run_agent = get_tool("run_agent")
    assert run_agent is not None

    with patch.object(
        run_agent, "execute", new=AsyncMock(return_value="it ran")
    ) as execute_mock:
        result = await execute_tool(
            tool_name="run_agent",
            parameters={},
            user_id="alice",
            session=session,
            tool_call_id="call-1",
            disabled_groups=(),
            disabled_tools=(
                kickoff_turn_disabled_tools()
                if is_expert_kickoff_turn(session)
                else frozenset()
            ),
        )

    execute_mock.assert_awaited_once()
    assert result == "it ran"
