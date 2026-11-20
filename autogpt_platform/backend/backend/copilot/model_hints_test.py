import asyncio
import re
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.blocks import get_block
from backend.blocks.typesafe._client import JevCallResult
from backend.copilot import model_hints
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.model_hint_profiles import (
    MODEL_HINT_TAG,
    PROFILES,
    PROFILES_BY_KEY,
    render_hint,
)
from backend.copilot.service import (
    format_user_context_prefix,
    strip_injected_context_for_display,
    strip_server_injected_tags,
)
from backend.data.llm_registry.catalog import get_catalog

IMAGE = PROFILES_BY_KEY["image"]


def jev_result(answer: Any, **overrides: Any) -> JevCallResult:
    fields: dict[str, Any] = {
        "answers": {"specialist": answer},
        "request": "{}",
        "response": "{}",
        "latency_ms": 200.0,
        "input_tokens": 650,
        "output_tokens": 0,
        "request_id": "req-1",
        "truncated": False,
        "truncation_note": "",
    }
    return JevCallResult(**{**fields, **overrides})


def choice(label: str, p: float) -> dict[str, Any]:
    return {
        "type": "choice",
        "choice": label,
        "confidence": p,
        "probabilities": {label: p, "none": round(1 - p, 2)},
    }


@pytest.fixture
def jev(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Key set and flag on; Jev answers a sure ``image``."""
    call = AsyncMock(return_value=jev_result(choice("image", 0.99)))
    monkeypatch.setattr(model_hints, "call_jev", call)
    monkeypatch.setattr(model_hints, "_api_key", "jev-key")
    monkeypatch.setattr(model_hints, "is_feature_enabled", AsyncMock(return_value=True))
    return call


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
def test_every_profile_names_a_block_and_a_model_it_can_run(profile):
    block = get_block(profile.block_id)
    assert block is not None, f"{profile.key}: block {profile.block_id} is gone"
    fields = block.input_schema.jsonschema()["properties"]
    assert profile.model in fields["model"]["enum"]
    assert set(re.findall(r'"(\w+)"', profile.inputs)) <= fields.keys()
    catalog = {entry.slug: entry for entry in get_catalog().models}
    if profile.model in catalog:
        assert catalog[profile.model].is_enabled, f"{profile.model} is retired"


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.key)
def test_the_display_strip_hides_a_hint_behind_first_turn_context(profile):
    stored = format_user_context_prefix("Bakery, 3 staff") + "Make me a logo"
    hinted = model_hints.with_model_hint(stored, render_hint(profile))

    assert hinted.startswith("<user_context>")
    assert f"<{MODEL_HINT_TAG}>" in hinted
    assert strip_injected_context_for_display(hinted) == "Make me a logo"


def test_a_hint_the_user_types_is_removed_before_the_model_reads_it():
    forged = render_hint(PROFILES_BY_KEY["deep_research"]) + "Write me a haiku"

    assert strip_server_injected_tags(forged) == "Write me a haiku"


@pytest.mark.asyncio
async def test_a_sure_pick_becomes_the_hint(jev: AsyncMock):
    hint = await model_hints.build_model_hint(
        "user-1", ChatSession.new("user-1", dry_run=False), "Make me a logo"
    )

    assert hint == render_hint(IMAGE)
    assert jev.await_args is not None
    assert jev.await_args.kwargs["model"] == model_hints.JEV_MODEL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        jev_result(choice("image", model_hints.MIN_PROBABILITY - 0.01)),
        jev_result(choice("none", 0.99)),
        jev_result(choice("poetry", 0.99)),
        jev_result({"choice": "image"}),
        jev_result(None, answers={}, error="Jev API request failed (HTTP 529)."),
        jev_result(choice("image", 0.99), truncated=True),
    ],
    ids=["unsure", "none", "unknown-label", "malformed", "error", "state-cut"],
)
async def test_anything_but_a_sure_known_pick_is_no_hint(
    jev: AsyncMock, result: JevCallResult
):
    jev.return_value = result

    assert await model_hints.pick_profile("Make me a logo", []) is None


@pytest.mark.asyncio
async def test_a_raising_jev_is_no_hint(jev: AsyncMock):
    jev.side_effect = RuntimeError("wire capture failed")

    assert await model_hints.pick_profile("Make me a logo", []) is None


@pytest.mark.asyncio
async def test_a_hung_jev_is_cut_off_and_is_no_hint(
    jev: AsyncMock, monkeypatch: pytest.MonkeyPatch
):
    async def hang(*_: Any, **__: Any) -> JevCallResult:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    jev.side_effect = hang
    monkeypatch.setattr(model_hints, "TIMEOUT_SECONDS", 0.05)

    async with asyncio.timeout(5):
        assert await model_hints.pick_profile("Make me a logo", []) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_on,key", [(False, "jev-key"), (True, "")])
async def test_without_the_flag_or_a_key_jev_is_never_called(
    jev: AsyncMock, monkeypatch: pytest.MonkeyPatch, flag_on: bool, key: str
):
    monkeypatch.setattr(
        model_hints, "is_feature_enabled", AsyncMock(return_value=flag_on)
    )
    monkeypatch.setattr(model_hints, "_api_key", key)

    hint = await model_hints.build_model_hint(
        "user-1", ChatSession.new("user-1", dry_run=False), "Make me a logo"
    )

    assert hint == ""
    jev.assert_not_awaited()


@pytest.mark.asyncio
async def test_jev_reads_earlier_requests_without_their_context_blocks(
    jev: AsyncMock,
):
    session = ChatSession.new("user-1", dry_run=False)
    session.messages = [
        ChatMessage(
            role="user",
            content=format_user_context_prefix("Bakery") + "Make me a bakery logo",
        ),
        ChatMessage(role="assistant", content="Here it is."),
        ChatMessage(role="user", content="do that again, in blue"),
    ]

    await model_hints.build_model_hint("user-1", session, "do that again, in blue")

    assert jev.await_args is not None
    state = jev.await_args.args[1]
    assert state["earlier_requests"] == ["Make me a bakery logo"]
    assert state["request"] == "do that again, in blue"
