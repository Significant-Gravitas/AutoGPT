"""Tests for the OpenRouter Broadcast attribution fields."""

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.copilot import llm_attribution
from backend.copilot.llm_attribution import (
    is_impersonated_request,
    is_turn_impersonated,
    openrouter_attribution,
    set_turn_impersonated,
)
from backend.util.settings import AppEnvironment


@pytest.fixture(autouse=True)
def _not_impersonated():
    set_turn_impersonated(False)
    yield
    set_turn_impersonated(False)


def test_attribution_names_the_user_chat_and_environment(mocker):
    mocker.patch.object(
        llm_attribution.settings.config, "app_env", AppEnvironment.DEVELOPMENT
    )

    body = openrouter_attribution(
        user_id="user-1", session_id="sess-1", generation_name="copilot-title"
    )

    assert body == {
        "user": "user-1",
        "session_id": "sess-1",
        "trace": {
            "generation_name": "copilot-title",
            "environment": "dev",
            "impersonated": False,
        },
    }


def test_attribution_without_a_user_or_chat_still_tags_the_environment():
    body = openrouter_attribution(
        user_id=None, session_id=None, generation_name="copilot-baseline"
    )

    assert set(body) == {"trace"}
    assert body["trace"]["generation_name"] == "copilot-baseline"


def test_attribution_marks_an_impersonated_turn():
    set_turn_impersonated(True)

    body = openrouter_attribution(
        user_id="user-1", session_id="sess-1", generation_name="copilot-sdk"
    )

    assert body["trace"]["impersonated"] is True


@pytest.mark.asyncio
async def test_impersonation_does_not_leak_into_another_turn():
    async def turn(impersonated: bool) -> bool:
        set_turn_impersonated(impersonated)
        await asyncio.sleep(0)
        return is_turn_impersonated()

    admin, user = await asyncio.gather(
        asyncio.create_task(turn(True)), asyncio.create_task(turn(False))
    )

    assert (admin, user) == (True, False)


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"X-Act-As-User-Id": "target-user"}, True),
        ({"X-Act-As-User-Id": "   "}, False),
        ({}, False),
    ],
)
def test_is_impersonated_request(headers: dict[str, str], expected: bool):
    request = MagicMock()
    request.headers = headers

    assert is_impersonated_request(request) is expected
