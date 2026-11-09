"""Tests for ``request_work_check`` and ``flag_mistake``.

The database and PostHog are mocked at the boundary;
``backend/data/feedback_prompts_test.py`` proves the limits against Postgres.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock

from backend.api.features.feedback_prompts import analytics
from backend.copilot.capabilities.sources import EAGER_CORE
from backend.copilot.model import ChatSession, ChatSessionOrigin
from backend.copilot.tools import TOOL_GROUPS, TOOL_REGISTRY
from backend.copilot.tools import feedback_prompts as feedback_tools
from backend.copilot.tools.feedback_prompts import FlagMistakeTool, RequestWorkCheckTool
from backend.copilot.tools.models import ErrorResponse, FeedbackPromptToolResponse
from backend.util.posthog_events import PostHogEvent

from ._test_data import make_session

USER_ID = "3b1f0e2a-7c4d-4e5f-8a9b-0c1d2e3f4a5b"
PROMPT_ID = "c3a1b2d4-e5f6-4789-8abc-def012345678"
WHAT = "I used last year's numbers."


@pytest.fixture
def db(mocker: pytest_mock.MockerFixture) -> MagicMock:
    database = MagicMock()
    database.claim_feedback_prompt = AsyncMock(return_value=PROMPT_ID)
    database.count_feedback_prompts = AsyncMock(return_value=1)
    mocker.patch.object(feedback_tools, "feedback_prompts_db", return_value=database)
    return database


@pytest.fixture
def shown(mocker: pytest_mock.MockerFixture) -> MagicMock:
    return mocker.patch.object(analytics, "capture")


@pytest.fixture
def flagged(mocker: pytest_mock.MockerFixture) -> MagicMock:
    return mocker.patch.object(feedback_tools, "capture")


@pytest.fixture
def web_chat() -> ChatSession:
    return make_session(USER_ID)


def _on_a_chat_platform(session: ChatSession) -> None:
    session.metadata.source_platform = "slack"


def _from_automation(session: ChatSession) -> None:
    session.metadata.origin = "automation"


def _delegated(session: ChatSession) -> None:
    session.metadata.delegated_by_session_id = "parent-session"


NO_CARD_CHATS = [_on_a_chat_platform, _from_automation, _delegated]


async def _work_check(session: ChatSession) -> FeedbackPromptToolResponse:
    result = await RequestWorkCheckTool()._execute(USER_ID, session)
    assert isinstance(result, FeedbackPromptToolResponse)
    return result


async def _flag(session: ChatSession) -> FeedbackPromptToolResponse:
    result = await FlagMistakeTool()._execute(USER_ID, session, what_went_wrong=WHAT)
    assert isinstance(result, FeedbackPromptToolResponse)
    return result


async def test_work_check_claims_and_returns_the_card(
    db: MagicMock, shown: MagicMock, web_chat: ChatSession
) -> None:
    result = await _work_check(web_chat)

    assert result.kind == "work_check"
    assert result.prompt_id == PROMPT_ID
    assert result.is_first is True
    db.claim_feedback_prompt.assert_awaited_once_with(
        user_id=USER_ID, kind="work_check", session_id=web_chat.session_id
    )
    shown.assert_called_once_with(
        USER_ID,
        PostHogEvent.FEEDBACK_PROMPT_SHOWN,
        {"prompt_id": PROMPT_ID, "kind": "work_check"},
    )


async def test_later_work_check_is_not_first(
    db: MagicMock, shown: MagicMock, web_chat: ChatSession
) -> None:
    db.count_feedback_prompts.return_value = 2

    assert (await _work_check(web_chat)).is_first is False


async def test_refused_work_check_tells_the_model_not_to_mention_it(
    db: MagicMock, shown: MagicMock, web_chat: ChatSession
) -> None:
    db.claim_feedback_prompt.return_value = None

    result = await _work_check(web_chat)

    assert result.prompt_id is None
    assert result.is_first is False
    assert "Do not mention" in result.message
    shown.assert_not_called()


async def test_a_failed_claim_is_a_refusal(
    db: MagicMock, shown: MagicMock, web_chat: ChatSession
) -> None:
    db.claim_feedback_prompt.side_effect = RuntimeError("db down")

    assert (await _work_check(web_chat)).prompt_id is None
    shown.assert_not_called()


@pytest.mark.parametrize(
    ("origin", "shows_card"),
    [("interactive", True), (None, True), ("automation", False)],
    ids=["interactive", "from-before-origin-existed", "automation"],
)
async def test_work_check_follows_the_chats_origin(
    db: MagicMock,
    shown: MagicMock,
    web_chat: ChatSession,
    origin: ChatSessionOrigin | None,
    shows_card: bool,
) -> None:
    web_chat.metadata.origin = origin

    result = await _work_check(web_chat)

    assert (result.prompt_id is not None) is shows_card
    assert db.claim_feedback_prompt.await_count == int(shows_card)


@pytest.mark.parametrize("arrange", NO_CARD_CHATS)
async def test_work_check_claims_nothing_where_no_card_can_show(
    db: MagicMock, shown: MagicMock, web_chat: ChatSession, arrange
) -> None:
    arrange(web_chat)

    assert (await _work_check(web_chat)).prompt_id is None
    db.claim_feedback_prompt.assert_not_awaited()
    shown.assert_not_called()


async def test_flag_mistake_in_a_web_chat_returns_the_card(
    db: MagicMock, shown: MagicMock, flagged: MagicMock, web_chat: ChatSession
) -> None:
    result = await _flag(web_chat)

    assert result.kind == "otto_mistake"
    assert result.prompt_id == PROMPT_ID
    db.claim_feedback_prompt.assert_awaited_once_with(
        user_id=USER_ID, kind="otto_mistake", session_id=web_chat.session_id
    )
    flagged.assert_called_once()
    assert flagged.call_args.args[1] == PostHogEvent.OTTO_MISTAKE_FLAGGED
    shown.assert_called_once()


@pytest.mark.parametrize("arrange", NO_CARD_CHATS)
async def test_flag_mistake_without_a_card_still_takes_the_slot_and_sends(
    db: MagicMock, shown: MagicMock, flagged: MagicMock, web_chat: ChatSession, arrange
) -> None:
    arrange(web_chat)

    result = await _flag(web_chat)

    assert result.prompt_id is None
    db.claim_feedback_prompt.assert_awaited_once()
    flagged.assert_called_once()
    user_id, event, properties = flagged.call_args.args
    assert user_id == USER_ID
    assert event == PostHogEvent.OTTO_MISTAKE_FLAGGED
    assert properties["$ai_session_id"] == web_chat.session_id
    shown.assert_not_called()


async def test_second_flag_in_a_chat_sends_nothing(
    db: MagicMock, shown: MagicMock, flagged: MagicMock, web_chat: ChatSession
) -> None:
    db.claim_feedback_prompt.return_value = None

    assert (await _flag(web_chat)).prompt_id is None
    flagged.assert_not_called()
    shown.assert_not_called()


async def test_flag_mistake_needs_a_description(
    db: MagicMock, flagged: MagicMock, web_chat: ChatSession
) -> None:
    result = await FlagMistakeTool()._execute(USER_ID, web_chat, what_went_wrong="  ")

    assert isinstance(result, ErrorResponse)
    db.claim_feedback_prompt.assert_not_awaited()
    flagged.assert_not_called()


async def test_impersonated_turn_neither_claims_nor_sends(
    mocker: pytest_mock.MockerFixture,
    db: MagicMock,
    shown: MagicMock,
    flagged: MagicMock,
    web_chat: ChatSession,
) -> None:
    mocker.patch.object(feedback_tools, "is_turn_impersonated", return_value=True)

    assert (await _work_check(web_chat)).prompt_id is None
    assert (await _flag(web_chat)).prompt_id is None
    db.claim_feedback_prompt.assert_not_awaited()
    shown.assert_not_called()
    flagged.assert_not_called()


@pytest.mark.parametrize("name", ["request_work_check", "flag_mistake"])
def test_both_tools_work_in_expert_chats_and_render_their_cards(name: str) -> None:
    assert name in TOOL_REGISTRY
    assert name in EAGER_CORE
    assert name not in TOOL_GROUPS


async def test_scheduled_followup_cannot_use_the_weekly_work_check_slot(
    mocker, db, shown, web_chat
):
    mocker.patch.object(feedback_tools, "is_unattended_turn", return_value=True)
    assert (await _work_check(web_chat)).prompt_id is None
    db.claim_feedback_prompt.assert_not_awaited()
    shown.assert_not_called()


async def test_mistake_event_does_not_export_conversation_content(
    db, flagged, web_chat
):
    await FlagMistakeTool()._execute(
        USER_ID, web_chat, what_went_wrong="Private Gmail content"
    )
    assert "Private Gmail content" not in str(flagged.call_args)
