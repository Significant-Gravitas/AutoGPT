"""Names in the transcript of record: only a misheard name is replaced, by the
name as both sides of the call's captions wrote it, and nothing else moves."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.voice_relay.names import apply_replacements, reconcile_names

TRANSCRIPT = (
    "I run Fern Brook Gardens and the books are in Zero.\nWe use WhatsApp groups."
)
USER = "I run Fernbrook Gardens and the books are in Xero. We use WhatsApp groups."
OTTO = "So Fernbrook Gardens keeps its books in Xero, and the crew is on WhatsApp."


def _apply(*pairs: tuple[str, str]) -> tuple[str, list]:
    return apply_replacements(
        TRANSCRIPT,
        [{"from": old, "to": new} for old, new in pairs],
        user=USER,
        otto=OTTO,
    )


def test_a_misheard_name_both_sides_wrote_is_put_right():
    fixed, made = _apply(("Zero", "Xero"))

    assert made == [("Zero", "Xero")]
    assert fixed == TRANSCRIPT.replace("Zero", "Xero")


def test_a_name_that_adds_or_drops_a_word_is_refused():
    """ "WhatsApp groups" is what the person said: shortening it would drop
    their word."""
    fixed, made = _apply(("WhatsApp groups", "WhatsApp"), ("Fern Brook", "Fernbrook"))

    assert made == []
    assert fixed == TRANSCRIPT


def test_a_name_only_one_side_of_the_call_wrote_is_refused():
    fixed, made = apply_replacements(
        "Our system is Surner.",
        [{"from": "Surner", "to": "Sonder"}],
        user="Our system is Sonder.",
        otto="You said your system is Cerner.",
    )

    assert made == []
    assert fixed == "Our system is Surner."


@pytest.mark.parametrize(
    "old,new",
    [
        ("books", "Xero"),  # not alike in spelling or sound
        ("Zero", "xero"),  # not written as a name
        ("Nowhere", "Xero"),  # not in the transcript
        ("Zero", "Quickbooks"),  # not in the captions
    ],
)
def test_anything_but_a_close_name_from_the_captions_is_refused(old, new):
    fixed, made = _apply((old, new))

    assert made == []
    assert fixed == TRANSCRIPT


@pytest.mark.parametrize("proposals", [None, "not a list", [1, "x", {"to": "Xero"}]])
def test_an_unreadable_proposal_changes_nothing(proposals):
    assert apply_replacements(TRANSCRIPT, proposals, user=USER, otto=OTTO) == (
        TRANSCRIPT,
        [],
    )


def test_only_the_whole_word_is_replaced():
    fixed, made = apply_replacements(
        "Zero and Zeros.",
        [{"from": "Zero", "to": "Xero"}],
        user="Xero",
        otto="Xero",
    )

    assert fixed == "Xero and Zeros."
    assert made == [("Zero", "Xero")]


@pytest.mark.asyncio
async def test_the_read_returns_the_transcript_put_right_and_its_cost():
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            text=json.dumps({"replacements": [{"from": "Zero", "to": "Xero"}]}),
            usage_metadata=SimpleNamespace(
                prompt_token_count=400,
                candidates_token_count=20,
                thoughts_token_count=0,
            ),
        )
    )

    read = await reconcile_names(
        client, model="m", transcript=TRANSCRIPT, user_captions=USER, otto_captions=OTTO
    )

    assert read.transcript == TRANSCRIPT.replace("Zero", "Xero")
    assert read.replaced == [("Zero", "Xero")]
    assert (read.tokens_in, read.tokens_out) == (400, 20)
