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

    content = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
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

    content = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert content.startswith(intro._LOCAL_PROMPT[:40])
    assert greeting == GENERATION["greeting"]
