"""Tests for the OpenRouter Broadcast attribution fields."""

import asyncio
import contextvars
from unittest.mock import MagicMock

import pytest

from backend.copilot import llm_attribution
from backend.copilot.llm_attribution import (
    is_impersonated_request,
    is_turn_impersonated,
    openrouter_attribution,
    record_turn_prompt,
    set_turn_impersonated,
    set_turn_origin,
    set_turn_trace_id,
    start_turn_prompt,
    turn_prompt,
    turn_trace_id,
)
from backend.util.posthog_prompts import PromptRef
from backend.util.settings import AppEnvironment


@pytest.fixture(autouse=True)
def _outside_a_turn():
    set_turn_impersonated(False)
    set_turn_trace_id(None)
    set_turn_origin(None)
    yield
    set_turn_impersonated(False)
    set_turn_trace_id(None)
    set_turn_origin(None)


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


@pytest.mark.parametrize("origin", ["chat", "scheduled", "spawned", "system"])
def test_attribution_says_what_started_the_turn(origin: str):
    set_turn_origin(origin)

    body = openrouter_attribution(
        user_id="user-1", session_id="sess-1", generation_name="copilot-sdk"
    )

    assert body["trace"]["turn_origin"] == origin


def test_attribution_outside_a_turn_sends_no_turn_origin():
    body = openrouter_attribution(
        user_id="user-1", session_id="sess-1", generation_name="copilot-title"
    )

    assert "turn_origin" not in body["trace"]


@pytest.mark.asyncio
async def test_concurrent_turns_each_send_their_own_origin():
    async def turn(origin: str) -> str | None:
        set_turn_origin(origin)
        await asyncio.sleep(0)
        return openrouter_attribution(
            user_id="user-1", session_id="sess-1", generation_name="copilot-sdk"
        )["trace"].get("turn_origin")

    person, schedule = await asyncio.gather(
        asyncio.create_task(turn("chat")), asyncio.create_task(turn("scheduled"))
    )

    assert (person, schedule) == ("chat", "scheduled")


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


def _in_a_turn_on(ref: PromptRef, generation_name) -> dict:
    """The attribution a call makes in a turn that runs on *ref*. Run in a copy
    of the context, so the turn's prompt record ends with the test."""

    def call() -> dict:
        start_turn_prompt()
        record_turn_prompt(ref)
        return openrouter_attribution(
            user_id="user-1", session_id="sess-1", generation_name=generation_name
        )

    return contextvars.copy_context().run(call)


@pytest.mark.parametrize("generation_name", ["copilot-sdk", "copilot-baseline"])
def test_a_turns_calls_name_the_system_prompt_it_runs_on(generation_name):
    body = _in_a_turn_on(PromptRef(name="copilot-prompt", version=7), generation_name)

    assert body["trace"]["prompt_name"] == "copilot-prompt"
    assert body["trace"]["prompt_version"] == 7
    assert body["trace"]["prompt_source"] == "posthog"


def test_a_turn_on_the_copy_in_code_names_no_version():
    body = _in_a_turn_on(PromptRef(name="copilot-prompt"), "copilot-sdk")

    assert body["trace"]["prompt_name"] == "copilot-prompt"
    assert body["trace"]["prompt_source"] == "code"
    assert "prompt_version" not in body["trace"]


def test_the_title_call_names_no_prompt():
    """The title is written with its own instructions, not the system prompt."""
    body = _in_a_turn_on(PromptRef(name="copilot-prompt", version=7), "copilot-title")

    assert not {"prompt_name", "prompt_version", "prompt_source"} & set(body["trace"])


def test_a_prompt_recorded_outside_a_turn_is_dropped():
    def call() -> PromptRef | None:
        record_turn_prompt(PromptRef(name="copilot-prompt", version=7))
        return turn_prompt()

    assert contextvars.copy_context().run(call) is None


@pytest.mark.asyncio
async def test_a_prompt_recorded_in_a_task_the_turn_started_reaches_the_turn():
    """The engines build the prompt under ``asyncio.gather``, in a child task
    whose own context changes never reach the turn's."""

    async def build() -> None:
        record_turn_prompt(PromptRef(name="copilot-prompt", version=7))

    async def turn() -> PromptRef | None:
        start_turn_prompt()
        await asyncio.gather(build())
        return turn_prompt()

    assert await asyncio.create_task(turn()) == PromptRef(
        name="copilot-prompt", version=7
    )
