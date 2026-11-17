"""Tests for picking Otto's fixed first line with Jev."""

from typing import get_args
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump import otto_reply
from backend.api.features.onboarding_dump.models import OttoReplyKind
from backend.blocks.typesafe._client import JevCallResult

# Pinned here so a kind added or dropped in one place fails until the
# frontend's lines (``insufficient-copy.ts``) are changed to match.
PINNED_KINDS = {
    "mic_check",
    "greeting",
    "unsure",
    "reluctant",
    "what_to_say",
    "wants_to_skip",
    "about_otto",
    "busy",
    "annoyed",
    "some_work",
    "garbled",
    "test",
    "other",
}


def _result(answer: dict | None, *, truncated: bool = False) -> JevCallResult:
    return JevCallResult(
        answers={} if answer is None else {"kind": answer},
        request="",
        response=None,
        latency_ms=1.0,
        input_tokens=None,
        output_tokens=None,
        request_id="",
        truncated=truncated,
        truncation_note="",
    )


@pytest.fixture
def jev(mocker: MockerFixture):
    mocker.patch.object(otto_reply, "_api_key", "test-key")

    def install(**kwargs) -> AsyncMock:
        mock = AsyncMock(**kwargs)
        mocker.patch.object(otto_reply, "call_jev", new=mock)
        return mock

    return install


def test_the_kinds_are_pinned():
    assert set(get_args(OttoReplyKind)) == PINNED_KINDS
    assert set(otto_reply.OTTO_REPLY_KINDS) == PINNED_KINDS


def test_each_mode_offers_only_kinds_that_fit_it():
    # A keyboard can't be heard, and a voice take isn't "a test".
    assert "mic_check" not in otto_reply.TYPED_CRITERIA
    assert "garbled" not in otto_reply.TYPED_CRITERIA
    assert "test" not in otto_reply.VOICE_CRITERIA
    assert (
        set(otto_reply.VOICE_CRITERIA) | set(otto_reply.TYPED_CRITERIA) == PINNED_KINDS
    )
    for criteria in (otto_reply.VOICE_CRITERIA, otto_reply.TYPED_CRITERIA):
        assert "other" in criteria
        assert all(criteria.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("typed", [False, True])
async def test_a_sure_pick_is_returned(jev, typed: bool):
    call = jev(
        return_value=_result(
            {"choice": "wants_to_skip", "probabilities": {"wants_to_skip": 0.9}}
        )
    )

    kind = await otto_reply.pick_reply_kind("  Can I skip this?  ", typed=typed)

    assert kind == "wants_to_skip"
    state, questions = call.await_args.args[1], call.await_args.args[2]
    assert state["said"] == "Can I skip this?"
    assert ("typed" in state["rubric"]) is typed
    criteria = questions["kind"].criteria
    expected = otto_reply.TYPED_CRITERIA if typed else otto_reply.VOICE_CRITERIA
    assert set(criteria) == set(expected)


@pytest.mark.asyncio
async def test_what_was_said_is_clipped(jev):
    call = jev(return_value=_result({"choice": "other", "confidence": 0.8}))

    await otto_reply.pick_reply_kind("word " * 500, typed=False)

    said = call.await_args.args[1]["said"]
    assert len(said) <= otto_reply.MAX_SAID_CHARS


@pytest.mark.asyncio
async def test_confidence_stands_in_for_missing_probabilities(jev):
    jev(return_value=_result({"choice": "busy", "confidence": 0.7}))

    assert await otto_reply.pick_reply_kind("One sec.", typed=False) == "busy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer,typed",
    [
        ({"choice": "busy", "probabilities": {"busy": 0.3}}, False),  # unsure
        ({"choice": "busy"}, False),  # no probability at all
        ({"choice": "write_a_poem", "confidence": 0.99}, False),  # not a kind
        ({"choice": "test", "confidence": 0.99}, False),  # typed-only kind
        ({"choice": "mic_check", "confidence": 0.99}, True),  # voice-only kind
        ({"choice": 3, "confidence": 0.99}, False),  # not text
        ({"pick": "mic_check"}, False),  # not the shape asked for
        (None, False),  # no answer
    ],
)
async def test_an_unusable_pick_leaves_the_default_line(jev, answer, typed: bool):
    jev(return_value=_result(answer))

    assert await otto_reply.pick_reply_kind("Hello?", typed=typed) is None


@pytest.mark.asyncio
async def test_a_cut_state_leaves_the_default_line(jev):
    jev(return_value=_result({"choice": "other", "confidence": 1.0}, truncated=True))

    assert await otto_reply.pick_reply_kind("Hello?", typed=False) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(), RuntimeError("boom")])
async def test_a_jev_failure_leaves_the_default_line(jev, error: Exception):
    jev(side_effect=error)

    assert await otto_reply.pick_reply_kind("Hello?", typed=False) is None


@pytest.mark.asyncio
async def test_no_key_or_no_words_never_calls_jev(jev, mocker: MockerFixture):
    call = jev(return_value=_result({"choice": "other", "confidence": 1.0}))

    assert await otto_reply.pick_reply_kind("   ", typed=False) is None
    mocker.patch.object(otto_reply, "_api_key", "")
    assert await otto_reply.pick_reply_kind("Hello?", typed=False) is None
    call.assert_not_awaited()
