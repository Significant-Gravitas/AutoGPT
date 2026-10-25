"""Tests for the greeting generation's prompt assembly.

The greeting promises the user specific automations, so what the model is
told this platform can connect to is part of the contract — a greeting
built without the registry offers integrations we do not have.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import intro

TRANSCRIPT = "I run a bakery and want the weekly order emails handled."

GENERATION = {
    "greeting": "You mentioned the weekly order emails.",
    "prompts": [
        {
            "title": f"Automation number {i}",
            "prompt": "Do it.",
            "icon": "sparkle",
            "integrations": [],
        }
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
        "backend.api.features.onboarding_dump.intro.connectable_lines",
        return_value="- slack: Team chat\n- github",
    )

    greeting, prompts = await intro.generate_intro(TRANSCRIPT)

    content = client.chat.completions.create.await_args.kwargs["messages"][-1][
        "content"
    ]
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
        "backend.api.features.onboarding_dump.intro.connectable_lines",
        return_value="",
    )

    greeting, _ = await intro.generate_intro(TRANSCRIPT)

    content = client.chat.completions.create.await_args.kwargs["messages"][-1][
        "content"
    ]
    assert content.startswith(intro._LOCAL_PROMPT[:40])
    assert greeting == GENERATION["greeting"]


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


def _generation_with(prompts: list[dict]) -> SimpleNamespace:
    body = {"greeting": "You sell stickers.", "prompts": prompts}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))]
    )


@pytest.mark.asyncio
async def test_each_prompt_keeps_the_integrations_it_names(client: MagicMock):
    named = {"title": "List each sticker", "prompt": "Do it.", "icon": "sparkle"}
    client.chat.completions.create.return_value = _generation_with(
        [{**named, "integrations": ["printify", " Shopify ", 7, ""]}]
        + [{**named, "integrations": []}] * 4
    )

    _, prompts = await intro.generate_intro(TRANSCRIPT)

    assert [p.integrations for p in prompts] == [
        ["printify", "Shopify"],
        [],
        [],
        [],
        [],
    ]


@pytest.mark.asyncio
async def test_a_prompt_that_does_not_say_what_it_connects_to_is_never_kept(
    client: MagicMock,
):
    """Stored without its integrations, nothing could warn the user that a
    task needs a service we lack, so such a generation is asked for again."""
    named = {"title": "List each sticker on Big Cartel", "prompt": "Do it."}
    client.chat.completions.create.side_effect = [
        _generation_with([named] * 5),
        _generation_with([{**named, "integrations": ["Big Cartel"]}] * 5),
    ]

    _, prompts = await intro.generate_intro(TRANSCRIPT)

    assert client.chat.completions.create.await_count == 2
    assert [p.integrations for p in prompts] == [["Big Cartel"]] * 5


@pytest.mark.asyncio
async def test_without_integrations_twice_the_generic_prompts_are_used(
    client: MagicMock,
):
    named = {"title": "List each sticker on Big Cartel", "prompt": "Do it."}
    client.chat.completions.create.return_value = _generation_with([named] * 5)

    greeting, prompts = await intro.generate_intro(TRANSCRIPT)

    assert greeting == "You sell stickers."
    assert prompts == intro.fallback_prompts()
    assert all(p.integrations is not None for p in prompts)


@pytest.mark.parametrize("unreadable", [[7], [""], [None, "  "]])
def test_integrations_with_no_usable_name_say_nothing(unreadable: list):
    """Only an empty array means "no sign-in"; one whose every entry is
    unusable stood for some service, so the prompt is not kept."""
    named = {"title": "List each sticker", "prompt": "Do it."}
    assert intro._parse_prompts([{**named, "integrations": unreadable}]) == []
    assert intro._parse_prompts([{**named, "integrations": []}])[0].integrations == []


def test_the_generic_prompts_say_they_connect_to_nothing():
    assert [p.integrations for p in intro.fallback_prompts()] == [[]] * len(
        intro.fallback_prompts()
    )


@pytest.mark.asyncio
async def test_the_integrations_ask_does_not_depend_on_the_langfuse_copy(
    client: MagicMock, mocker: MockerFixture
):
    """Langfuse's copy of the instructions wins in prod and may predate the
    field, so the ask for it is sent from code, in the system message."""
    mocker.patch(
        "backend.api.features.onboarding_dump.intro._fetch_langfuse_prompt",
        new=AsyncMock(return_value="Older instructions.\n\nTranscript:\n"),
    )

    await intro.generate_intro(TRANSCRIPT)

    system, user = client.chat.completions.create.await_args.kwargs["messages"]
    assert system["role"] == "system"
    assert '"integrations"' in system["content"]
    assert "not on the list" in system["content"]
    assert user["content"].endswith(TRANSCRIPT)


@pytest.mark.asyncio
@pytest.mark.parametrize("icon", [["bell"], {"name": "bell"}])
async def test_a_malformed_icon_costs_only_the_icon(client: MagicMock, icon: object):
    """generate_intro never raises: an unhashable icon from the model falls
    back to the default instead of escaping the set lookup."""
    named = {"title": "List each sticker", "prompt": "Do it.", "integrations": []}
    client.chat.completions.create.return_value = _generation_with(
        [{**named, "icon": icon}] * 5
    )

    _, prompts = await intro.generate_intro(TRANSCRIPT)

    assert [p.icon for p in prompts] == [intro.DEFAULT_PROMPT_ICON] * 5
