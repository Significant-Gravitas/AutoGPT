"""The details the ending screens show, read from the model's answer and
cleaned in code."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.onboarding_interview_details import (
    SCHEMA,
    conversation_text,
    extract_details,
    parse_details,
)


def _answer(**fields) -> str:
    return json.dumps({"time_sinks": [], "tools": [], "topics": [], **fields})


def test_the_answer_is_cleaned_and_bounded():
    details = parse_details(
        _answer(
            business="  Landscaping\n crew ",
            tools=["QuickBooks", "QuickBooks", "", 7, *[f"tool {n}" for n in range(9)]],
            topics=["tools", "week", "not-a-topic", "work"],
        )
    )

    assert details is not None
    assert details.business == "Landscaping crew"
    assert details.tools[0] == "QuickBooks"
    assert len(details.tools) == 6
    # In the interview's order, and only the five it knows.
    assert details.topics == ["work", "week", "tools"]


def test_quoted_facts_in_an_answer_are_not_kept():
    details = parse_details(
        _answer(
            business="Landscaping",
            facts=[{"label": "Crew", "quote": "a six-person landscaping crew"}],
        )
    )

    assert details is not None
    assert "facts" not in details.model_dump()
    assert "words" not in details.model_dump()


@pytest.mark.parametrize("text", [None, "", "not json", "[1, 2]"])
def test_an_unreadable_answer_reads_as_none(text):
    assert parse_details(text) is None


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
async def test_a_read_asks_for_no_quoted_facts_and_says_what_it_cost():
    client = _client(_answer(business="Landscaping", first_task="Chase invoices"))
    lines = [("otto", "What do you do?"), ("user", "I mow lawns")]

    read = await extract_details(client, model="m", lines=lines)

    assert read.details is not None
    assert read.details.business == "Landscaping"
    assert read.details.first_task == "Chase invoices"
    assert (read.tokens_in, read.tokens_out) == (900, 150)
    kwargs = client.aio.models.generate_content.await_args.kwargs
    assert kwargs["config"].response_json_schema == SCHEMA
    assert "facts" not in SCHEMA["properties"]
    assert '"facts"' not in kwargs["contents"]
