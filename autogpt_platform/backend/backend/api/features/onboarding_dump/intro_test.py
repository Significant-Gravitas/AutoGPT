"""Tests for the greeting generation's prompt assembly.

The greeting promises the user specific automations, so what the model is
told this platform can connect to is part of the contract — a greeting
built without the registry offers integrations we do not have.
"""

import json
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import openai
import pytest
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import intro

TRANSCRIPT = "I run a bakery and want the weekly order emails handled."

GENERATION = {
    "greeting": "You mentioned the weekly order emails.",
    "prompts": [
        {"title": f"Automation number {i}", "prompt": "Do it.", "icon": "sparkle"}
        for i in range(5)
    ],
}


@pytest.fixture
def client(mocker: MockerFixture) -> MagicMock:
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps(GENERATION)))
        ]
    )
    fake = MagicMock()
    fake.chat.completions.create = AsyncMock(return_value=completion)
    mocker.patch(
        "backend.api.features.onboarding_dump.intro.get_openai_client",
        return_value=fake,
    )
    mocker.patch(
        "backend.api.features.onboarding_dump.intro._fetch_langfuse_prompt",
        new=AsyncMock(return_value=None),
    )
    return fake


@pytest.mark.asyncio
async def test_greeting_prompt_carries_the_provider_registry(
    client: MagicMock, mocker: MockerFixture
):
    mocker.patch(
        "backend.api.features.onboarding_dump.intro.known_providers",
        return_value={"slack": "Team chat", "github": None},
    )

    greeting, prompts = await intro.generate_intro(TRANSCRIPT)

    content = _sent(client, "user")
    assert "- slack: Team chat" in content
    assert "- github" in content
    # Registry first, then the instructions, then the transcript they end
    # with — the static half of the message stays a stable prefix.
    assert content.index("- slack") < content.index(TRANSCRIPT)
    assert greeting == GENERATION["greeting"]
    assert len(prompts) == 5


@pytest.mark.asyncio
async def test_greeting_prompt_survives_an_empty_registry(
    client: MagicMock, mocker: MockerFixture
):
    mocker.patch(
        "backend.api.features.onboarding_dump.intro.known_providers",
        return_value={},
    )

    greeting, _ = await intro.generate_intro(TRANSCRIPT)

    content = _sent(client, "user")
    assert content.startswith(intro._LOCAL_PROMPT[:40])
    assert greeting == GENERATION["greeting"]


@pytest.mark.asyncio
async def test_the_coverage_rule_rides_along_with_the_langfuse_copy(
    client: MagicMock, mocker: MockerFixture
):
    """The Langfuse copy wins in prod, so the rule cannot live in either copy."""
    mocker.patch(
        "backend.api.features.onboarding_dump.intro._fetch_langfuse_prompt",
        new=AsyncMock(return_value="Langfuse instructions.\nTranscript:\n"),
    )

    await intro.generate_intro(TRANSCRIPT)

    assert _sent(client, "system") == intro.COVERAGE_CONTRACT
    assert "Langfuse instructions." in _sent(client, "user")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "long_greeting, ending",
    [
        ("You have a problem with the oven and the ordering. " * 20, "."),
        # The only sentence end is too early to keep, so it stops at a word.
        ("Hi there. " + "your oven and your ordering " * 40, "…"),
    ],
    ids=["many sentences", "one early sentence"],
)
async def test_an_over_long_greeting_keeps_most_of_itself(
    client: MagicMock, long_greeting: str, ending: str
):
    client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
        {**GENERATION, "greeting": long_greeting}
    )

    greeting, _ = await intro.generate_intro(TRANSCRIPT)

    kept = greeting.removesuffix("…")
    assert intro.MAX_GREETING_CHARS // 2 < len(greeting) <= intro.MAX_GREETING_CHARS
    assert greeting.endswith(ending)
    assert long_greeting.startswith(kept) and long_greeting[len(kept)] == " "


@pytest.mark.asyncio
async def test_sonnet_writes_the_greeting_through_openrouter_s_reasoning_rule(
    client: MagicMock, mocker: MockerFixture
):
    """OpenRouter refuses Sonnet 5.5 a request with reasoning off, which handed
    every greeting to the backup from v0.11.2 on."""
    mocker.patch.object(intro, "_MODEL", "anthropic/claude-sonnet-5.5")
    client.chat.completions.create.side_effect = _openrouter

    greeting, prompts = await intro.generate_intro(TRANSCRIPT)

    assert greeting == "Written by anthropic/claude-sonnet-5.5."
    assert len(prompts) == 5
    models = [c.kwargs["model"] for c in client.chat.completions.create.await_args_list]
    assert models == ["anthropic/claude-sonnet-5.5"]


def test_both_copies_of_the_instructions_ask_for_the_same_length():
    """The fallback instructions must not cap the greeting below what the
    system message allows, and the clip must sit above both."""
    # The optional ``why`` clause under each prompt has its own, shorter cap.
    why_limit = str(intro.MAX_WHY_CHARS)
    for text in (intro._LOCAL_PROMPT, intro.COVERAGE_CONTRACT):
        lengths = re.findall(r"(\d+) characters", text)
        assert [n for n in lengths if n != why_limit] == [
            str(intro.GREETING_CHAR_LIMIT)
        ]
    assert intro.GREETING_CHAR_LIMIT < intro.MAX_GREETING_CHARS


def test_the_local_prompt_asks_for_the_head_of_ai_voice():
    """The greeting diagnoses and promises a team; it never casts it.

    The team section under the greeting is built from filtered template
    ids, so prose that names an expert can promise a colleague who was
    never recommended — or does not exist on the roster at all.
    """
    assert "Head of AI" in intro._LOCAL_PROMPT
    assert "NEVER name a specific expert" in intro._LOCAL_PROMPT
    # The JSON contract the parser depends on is unchanged.
    assert '"greeting"' in intro._LOCAL_PROMPT
    assert '"prompts"' in intro._LOCAL_PROMPT


@pytest.mark.asyncio
async def test_the_generation_contract_is_still_greeting_plus_prompts(
    client: MagicMock,
):
    greeting, suggested = await intro.generate_intro(TRANSCRIPT)

    assert greeting == GENERATION["greeting"]
    assert [p.title for p in suggested] == [
        item["title"] for item in GENERATION["prompts"]
    ]


def test_the_fallback_greeting_names_nobody():
    """It runs when the model gave us nothing, so it can claim nothing.

    Naming an expert here would be a hire the roster never made.
    """
    for transcript in ("", TRANSCRIPT):
        greeting, suggested = intro.fallback_intro(transcript)
        assert "Head of AI" in greeting
        assert not any(name in greeting for name in ("Maria", "Max", "Frankie"))
        assert suggested == intro.fallback_prompts()


def _sent(client: MagicMock, role: str) -> str:
    messages = client.chat.completions.create.await_args.kwargs["messages"]
    return next(m["content"] for m in messages if m["role"] == role)


async def _openrouter(**kwargs: Any) -> SimpleNamespace:
    """A completion the way OpenRouter answers it: Sonnet 5.5's endpoint
    refuses reasoning off, and every greeting names the model that wrote it."""
    reasoning = (kwargs.get("extra_body") or {}).get("reasoning") or {}
    if "claude-sonnet-5" in kwargs["model"] and reasoning.get("enabled") is False:
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        raise openai.BadRequestError(
            "Reasoning is mandatory for this endpoint and cannot be disabled.",
            response=httpx.Response(400, request=request),
            body=None,
        )
    data = {**GENERATION, "greeting": f"Written by {kwargs['model']}."}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))]
    )


def test_the_local_prompt_offers_an_optional_why_per_prompt():
    assert '"why" (optional)' in intro._LOCAL_PROMPT
    assert f"at most {intro.MAX_WHY_CHARS} characters" in intro._LOCAL_PROMPT


def test_a_prompt_keeps_its_why_and_one_without_still_parses():
    at_cap = "x" * intro.MAX_WHY_CHARS
    prompts = intro._parse_prompts(
        [
            {
                "title": "Rewrite listings",
                "prompt": "Do it.",
                "why": " Listings  eat your evenings. ",
            },
            {"title": "Find shops", "prompt": "Do it."},
            {"title": "Answer reviews", "prompt": "Do it.", "why": at_cap},
            {"title": "Track orders", "prompt": "Do it.", "why": at_cap + "x"},
            {"title": "Plan a drop", "prompt": "Do it.", "why": 7},
        ]
    )

    assert [p.why for p in prompts] == [
        "Listings eat your evenings.",
        None,
        at_cap,
        None,  # too long to read as one clause
        None,
    ]
    assert all(p.expert_template_id is None for p in prompts)


@pytest.mark.asyncio
async def test_a_failed_sonnet_call_gets_the_greeting_from_the_backup(
    client: MagicMock,
):
    # On a deploy where the Sonnet call fails, the user still gets tasks
    # written from what they said, not the role's starters.
    completion = client.chat.completions.create.return_value
    client.chat.completions.create.side_effect = [
        RuntimeError("model not available"),
        completion,
    ]

    greeting, prompts = await intro.generate_intro(TRANSCRIPT)

    assert greeting == GENERATION["greeting"]
    assert [p.title for p in prompts] == [p["title"] for p in GENERATION["prompts"]]
    calls = client.chat.completions.create.await_args_list
    assert [c.kwargs["model"] for c in calls] == [intro._MODEL, intro._BACKUP_MODEL]
    # The backup is the fast answer inside the plan step's wait: no thinking first.
    assert calls[1].kwargs["extra_body"] == {"reasoning": {"enabled": False}}


@pytest.mark.asyncio
async def test_when_both_models_fail_the_template_answers(client: MagicMock):
    client.chat.completions.create.side_effect = RuntimeError("gateway down")

    assert await intro.generate_intro(TRANSCRIPT) == intro.fallback_intro(TRANSCRIPT)
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed",
    [
        # A choice with no message, as OpenRouter sends with finish_reason
        # "error".
        SimpleNamespace(choices=[SimpleNamespace(message=None, finish_reason="error")]),
        # A 2xx HTML error page, which the SDK hands back as a plain string.
        "<html>Bad gateway</html>",
    ],
    ids=["no message", "html body"],
)
async def test_a_malformed_sonnet_reply_still_gets_the_backup_s_greeting(
    client: MagicMock, malformed: object
):
    completion = client.chat.completions.create.return_value
    client.chat.completions.create.side_effect = [malformed, completion]

    greeting, _ = await intro.generate_intro(TRANSCRIPT)

    assert greeting == GENERATION["greeting"]
    assert client.chat.completions.create.await_count == 2
