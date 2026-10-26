"""Tests for the greeting generation's prompt assembly.

The greeting promises the user specific automations, so what the model is
told this platform can connect to is part of the contract — a greeting
built without the registry offers integrations we do not have.
"""

import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.integrations import support
from backend.api.features.onboarding_dump import intro
from backend.api.features.onboarding_dump.providers import label_support

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
        "backend.api.features.onboarding_dump.intro.connectable_lines",
        return_value="",
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


def test_both_copies_of_the_instructions_ask_for_the_same_length():
    """The fallback instructions must not cap the greeting below what the
    system message allows, and the clip must sit above both."""
    for text in (intro._LOCAL_PROMPT, intro.COVERAGE_CONTRACT):
        assert re.findall(r"(\d+) characters", text) == [str(intro.GREETING_CHAR_LIMIT)]
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
async def test_a_model_that_ignores_integrations_keeps_the_users_own_tasks(
    client: MagicMock, mocker: MockerFixture
):
    """If the production model leaves the field out every time, the user
    keeps their own tasks rather than the generic set: they are asked for
    once more, kept as "not declared", and labelled from their text."""
    mocker.patch.object(
        support,
        "connectable_services",
        return_value=[support.ConnectableService(key="printify", name="Printify")],
    )
    named = {"title": "List each sticker on Shopify", "prompt": "Do it."}
    client.chat.completions.create.return_value = _generation_with([named] * 5)

    greeting, prompts = await intro.generate_intro(TRANSCRIPT)

    assert client.chat.completions.create.await_count == 2
    assert greeting == "You sell stickers."
    assert [p.title for p in prompts] == ["List each sticker on Shopify"] * 5
    assert all(p.integrations is None for p in prompts)
    assert [p.unsupported for p in label_support(prompts)] == [["Shopify"]] * 5


@pytest.mark.asyncio
async def test_prompts_that_declare_their_integrations_keep_them_beside_ones_that_do_not(
    client: MagicMock,
):
    named = {"title": "List each sticker", "prompt": "Do it."}
    client.chat.completions.create.return_value = _generation_with(
        [{**named, "integrations": ["printify"]}] * 3 + [named] * 2
    )

    _, prompts = await intro.generate_intro(TRANSCRIPT)

    assert [p.integrations for p in prompts] == [["printify"]] * 3 + [None] * 2


@pytest.mark.parametrize("unreadable", [[7], [""], [None, "  "]])
def test_integrations_with_no_usable_name_say_nothing(unreadable: list):
    """Only an empty array means "no sign-in"; one whose every entry is
    unusable stood for some service, so it counts as not declared."""
    named = {"title": "List each sticker", "prompt": "Do it."}
    [parsed] = intro._parse_prompts([{**named, "integrations": unreadable}])
    assert parsed.integrations is None
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

    messages = client.chat.completions.create.await_args.kwargs["messages"]
    systems = [m["content"] for m in messages if m["role"] == "system"]
    # Both code-owned rules ride along, each as written: coverage first.
    assert systems == [intro.COVERAGE_CONTRACT, intro.INTEGRATIONS_CONTRACT]
    assert '"integrations"' in systems[1] and "not on the list" in systems[1]
    assert _sent(client, "user").endswith(TRANSCRIPT)


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


def _attempt(greeting: str, prompt_count: int) -> SimpleNamespace:
    body = {
        "greeting": greeting,
        "prompts": [
            {"title": f"Task {i}", "prompt": "Do it.", "integrations": []}
            for i in range(prompt_count)
        ],
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))]
    )


@pytest.mark.asyncio
async def test_a_retry_without_a_greeting_keeps_the_first_attempts_greeting(
    client: MagicMock,
):
    """The first attempt's greeting was good but its prompts too few; a
    retry that comes back without a greeting must not cost the user it."""
    client.chat.completions.create.side_effect = [
        _attempt("You sell stickers.", 2),
        _attempt("", 5),
    ]

    greeting, prompts = await intro.generate_intro(TRANSCRIPT)

    assert greeting == "You sell stickers."
    assert prompts == intro.fallback_prompts()


@pytest.mark.asyncio
async def test_a_first_attempt_without_a_greeting_is_asked_for_again(
    client: MagicMock,
):
    client.chat.completions.create.side_effect = [
        _attempt("", 5),
        _attempt("You sell stickers.", 5),
    ]

    greeting, prompts = await intro.generate_intro(TRANSCRIPT)

    assert client.chat.completions.create.await_count == 2
    assert greeting == "You sell stickers."
    assert [p.title for p in prompts] == [f"Task {i}" for i in range(5)]


def _sent(client: MagicMock, role: str) -> str:
    messages = client.chat.completions.create.await_args.kwargs["messages"]
    return next(m["content"] for m in messages if m["role"] == role)
