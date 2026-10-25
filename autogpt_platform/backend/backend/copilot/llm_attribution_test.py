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
    set_turn_trace_id,
    turn_trace_id,
)
from backend.util.settings import AppEnvironment


@pytest.fixture(autouse=True)
def _outside_a_turn():
    set_turn_impersonated(False)
    set_turn_trace_id(None)
    yield
    set_turn_impersonated(False)
    set_turn_trace_id(None)


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


@pytest.mark.parametrize(
    ("sentry_release", "expected"),
    [("0123abcd\n", {"revision": "0123abcd"}), ("", {})],
    ids=["revision set", "no revision"],
)
def test_attribution_names_the_revision_that_made_the_call(
    mocker, sentry_release: str, expected: dict[str, str]
):
    mocker.patch.object(
        llm_attribution.settings.config, "sentry_release", sentry_release
    )

    body = openrouter_attribution(
        user_id=None, session_id=None, generation_name="copilot-sdk"
    )

    assert {k: v for k, v in body["trace"].items() if k == "revision"} == expected


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


def test_attribution_groups_a_turns_calls_under_its_trace():
    set_turn_trace_id("turn-1")

    body = openrouter_attribution(
        user_id="user-1", session_id="sess-1", generation_name="copilot-sdk"
    )

    assert body["trace"]["trace_id"] == "turn-1"


def test_attribution_outside_a_turn_sends_no_trace_id():
    body = openrouter_attribution(
        user_id="user-1", session_id="sess-1", generation_name="copilot-title"
    )

    assert "trace_id" not in body["trace"]


def test_a_blank_turn_id_sends_no_trace_id():
    set_turn_trace_id("")

    assert turn_trace_id() is None
    body = openrouter_attribution(
        user_id="user-1", session_id="sess-1", generation_name="copilot-sdk"
    )
    assert "trace_id" not in body["trace"]


@pytest.mark.asyncio
async def test_a_trace_id_does_not_leak_into_another_turn():
    async def turn(turn_id: str) -> str | None:
        set_turn_trace_id(turn_id)
        await asyncio.sleep(0)
        return openrouter_attribution(
            user_id="user-1", session_id="sess-1", generation_name="copilot-sdk"
        )["trace"]["trace_id"]

    first, second = await asyncio.gather(
        asyncio.create_task(turn("turn-1")), asyncio.create_task(turn("turn-2"))
    )

    assert (first, second) == ("turn-1", "turn-2")


@pytest.mark.asyncio
async def test_a_task_the_turn_starts_shares_its_trace():
    """The chat title is generated in a task the first turn starts, so its
    call lands in that turn's trace."""
    set_turn_trace_id("turn-1")

    async def title_call() -> str | None:
        return turn_trace_id()

    assert await asyncio.create_task(title_call()) == "turn-1"


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
