import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.api.features.experts import onboarding

EXPERT_ID = "expert-kepler"
SESSION_ID = "session-kickoff"


def test_route_requires_authentication():
    app = FastAPI()
    app.include_router(onboarding.router)
    response = TestClient(app).get(f"/{EXPERT_ID}/onboarding")
    assert response.status_code in (401, 403)


def test_route_returns_saved_questions_for_authenticated_owner(mocker):
    app = FastAPI()
    app.include_router(onboarding.router)
    app.dependency_overrides[onboarding.auth.get_user_id] = lambda: "user-1"
    expert = mocker.patch.object(
        onboarding.experts_db, "get_expert", return_value=mocker.Mock(is_archived=False)
    )
    mocker.patch.object(
        onboarding, "query_raw_with_schema", new=AsyncMock(return_value=[card()])
    )
    response = TestClient(app).get(f"/{EXPERT_ID}/onboarding")
    assert response.status_code == 200
    assert response.json() == {"session_id": SESSION_ID}
    expert.assert_awaited_once_with("user-1", EXPERT_ID)


def card(session_id: str = SESSION_ID, **overrides) -> onboarding.OnboardingMessage:
    return onboarding.OnboardingMessage(
        session_id=session_id,
        role="tool",
        content=json.dumps(
            {
                "type": "expert_onboarding",
                "message": "Setup",
                "expert_id": EXPERT_ID,
                "greeting": "I'm Kepler.",
                "steps": [
                    {"keyword": "topic", "question": "What topic?", "options": []},
                    {"keyword": "format", "question": "What format?", "options": []},
                ],
                **overrides,
            }
        ),
    )


def reply(content: str, session_id: str = SESSION_ID) -> onboarding.OnboardingMessage:
    return onboarding.OnboardingMessage(
        session_id=session_id, role="user", content=content
    )


@pytest.mark.parametrize(
    "reply_text, pending",
    [
        ("", True),
        ("Research this company", True),
        (onboarding.SKIP_MESSAGE, False),
        (
            "**Here are my answers:**\n\n> What topic?\n\nAI\n\n"
            "> What format?\n\nReport\n\nPlease proceed.",
            False,
        ),
        ("**Here are my answers:**\n\n> What topic?\n\nAI\n\nPlease proceed.", True),
        (
            "**Here are my answers:**\n\n> Another question?\n\nAI\n\nPlease proceed.",
            True,
        ),
        (
            "**Here are my answers:**\n\n> What topic?\n\n\n\n> What format?\n\nReport\n\nPlease proceed.",
            True,
        ),
    ],
)
def test_complete_answers_or_skip_from_another_chat_settle_setup(reply_text, pending):
    messages = [card(), reply(reply_text, session_id="session-other")]
    assert (onboarding.pending_onboarding(messages, EXPERT_ID) is not None) == pending


def test_points_at_the_chat_holding_the_card():
    messages = [reply("Kick off"), card(), reply("Hello", session_id="session-other")]
    assert onboarding.pending_onboarding(messages, EXPERT_ID) == SESSION_ID


def test_any_reply_in_the_cards_chat_settles_it():
    messages = [card(), reply("Research this company")]
    assert onboarding.pending_onboarding(messages, EXPERT_ID) is None


def test_a_card_still_waiting_in_an_older_chat_is_found():
    messages = [
        card(session_id="session-old"),
        card(),
        reply("Research this company"),
    ]
    assert onboarding.pending_onboarding(messages, EXPERT_ID) == "session-old"


def test_completion_applies_across_chats_and_later_cards():
    messages = [
        card(),
        reply(onboarding.SKIP_MESSAGE),
        card(session_id="session-other"),
    ]
    assert onboarding.pending_onboarding(messages, EXPERT_ID) is None


def test_ignores_invalid_cards_and_other_experts():
    messages = [
        onboarding.OnboardingMessage(
            session_id=SESSION_ID, role="tool", content="invalid json"
        ),
        card(expert_id="another-expert"),
        card(steps=[]),
        card(type="error"),
    ]
    assert onboarding.pending_onboarding(messages, EXPERT_ID) is None


@pytest.mark.asyncio
async def test_history_query_scopes_user_and_expert(mocker):
    mocker.patch.object(
        onboarding.experts_db, "get_expert", return_value=mocker.Mock(is_archived=False)
    )
    query = mocker.patch.object(
        onboarding, "query_raw_with_schema", new=AsyncMock(return_value=[card()])
    )
    result = await onboarding.get_expert_onboarding(EXPERT_ID, "user-1")
    assert result is not None
    assert query.call_args.args[1:] == ("user-1", EXPERT_ID, onboarding.SKIP_MESSAGE)
    assert query.call_args.kwargs["model"] is onboarding.OnboardingMessage
    assert 's."userId" = $1 AND s."expertId" = $2' in query.call_args.args[0]
    assert "OR m.role = 'user'" in query.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("archived", [False, True])
async def test_unavailable_experts_do_not_read_history(mocker, archived):
    mocker.patch.object(
        onboarding.experts_db,
        "get_expert",
        return_value=mocker.Mock(is_archived=True) if archived else None,
    )
    query = mocker.patch.object(onboarding, "query_raw_with_schema", new=AsyncMock())
    with pytest.raises(HTTPException) as error:
        await onboarding.get_expert_onboarding(EXPERT_ID, "user-1")
    assert error.value.status_code == 404
    query.assert_not_called()
