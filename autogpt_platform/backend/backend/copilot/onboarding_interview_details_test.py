"""The details the ending screens show: every fact tied to the person's own
words, checked in code."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.onboarding_interview_details import (
    LIVE_SCHEMA,
    SCHEMA,
    conversation_text,
    extract_details,
    parse_details,
    quote_in_words,
)

WORDS = (
    "I run Fernbrook Gardens, a six-person landscaping crew.\n"
    "My evenings go to quotes and chasing invoices in QuickBooks."
)


def _answer(**fields) -> str:
    return json.dumps(
        {"time_sinks": [], "tools": [], "facts": [], "topics": [], **fields}
    )


def test_a_quote_said_word_for_word_is_found_whatever_its_case_and_punctuation():
    assert quote_in_words("a six-person landscaping crew", WORDS)
    assert quote_in_words("My evenings go to QUOTES and chasing invoices", WORDS)
    assert quote_in_words("fernbrook gardens", WORDS)


def test_a_paraphrase_or_a_fragment_of_a_word_is_not_a_quote():
    assert not quote_in_words("I own a landscaping company", WORDS)
    assert not quote_in_words("brook Gard", WORDS)
    assert not quote_in_words("", WORDS)


def test_a_fact_whose_quote_the_person_never_said_is_dropped():
    details = parse_details(
        _answer(
            facts=[
                {"label": "Six people", "quote": "a six-person landscaping crew"},
                {"label": "Uses Jobber", "quote": "we schedule everything in Jobber"},
            ]
        ),
        WORDS,
    )

    assert details is not None
    assert [fact.label for fact in details.facts] == ["Six people"]


def test_the_answer_is_cleaned_and_bounded():
    details = parse_details(
        _answer(
            business="  Landscaping\n crew ",
            tools=["QuickBooks", "QuickBooks", "", 7, *[f"tool {n}" for n in range(9)]],
            topics=["tools", "week", "not-a-topic", "work"],
        ),
        WORDS,
    )

    assert details is not None
    assert details.business == "Landscaping crew"
    assert details.tools[0] == "QuickBooks"
    assert len(details.tools) == 6
    # In the interview's order, and only the five it knows.
    assert details.topics == ["work", "week", "tools"]


@pytest.mark.parametrize("text", [None, "", "not json", "[1, 2]"])
def test_an_unreadable_answer_reads_as_none(text):
    assert parse_details(text, WORDS) is None


def test_the_conversation_names_each_side_and_keeps_its_latest_part():
    text = conversation_text([("otto", "What do you do?"), ("user", " I  mow ")])

    assert text == "OTTO: What do you do?\nUSER: I mow"


def _client(answer: str) -> MagicMock:
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            text=answer,
            usage_metadata=SimpleNamespace(
                prompt_token_count=900,
                candidates_token_count=120,
                thoughts_token_count=30,
            ),
        )
    )
    return client


@pytest.mark.asyncio
async def test_a_final_read_carries_the_persons_lines_and_what_it_cost():
    client = _client(
        _answer(facts=[{"label": "Crew", "quote": "a six-person landscaping crew"}])
    )
    lines = [("otto", "What do you do?"), ("user", WORDS.splitlines()[0])]

    read = await extract_details(client, model="m", lines=lines)

    assert read.details is not None
    assert read.details.words == [WORDS.splitlines()[0]]
    assert len(read.details.facts) == 1
    assert (read.tokens_in, read.tokens_out) == (900, 150)
    config = client.aio.models.generate_content.await_args.kwargs["config"]
    assert config.response_json_schema == SCHEMA


@pytest.mark.asyncio
async def test_a_live_read_asks_for_no_facts_and_carries_no_lines():
    client = _client(_answer(business="Landscaping"))

    read = await extract_details(
        client, model="m", lines=[("user", "I mow lawns")], live=True
    )

    assert read.details is not None
    assert read.details.words == []
    kwargs = client.aio.models.generate_content.await_args.kwargs
    assert kwargs["config"].response_json_schema == LIVE_SCHEMA
    assert "facts" not in LIVE_SCHEMA["properties"]
    assert '"facts"' not in kwargs["contents"]
