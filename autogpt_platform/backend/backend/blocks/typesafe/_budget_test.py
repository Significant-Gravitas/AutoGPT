import json
import os

import pytest
from typesafe_sdk import Choice, Noul
from typesafe_sdk._core.json import serialize
from typesafe_sdk.constants import DEFAULT_MODEL, DEFAULT_MODEL_ENV

from backend.blocks.typesafe._budget import MAX_REQUEST_BYTES, prepare_state


def request_size(state: str, questions: dict) -> int:
    return len(
        serialize(
            {
                "state": state,
                "model": os.environ.get(DEFAULT_MODEL_ENV, "").strip() or DEFAULT_MODEL,
                "questions": questions,
            }
        )
    )


def test_json_state_is_compact_and_plain_text_is_unchanged():
    questions = {"q": Noul(instructions="Is it relevant?")}
    structured = prepare_state({"label": "é", "items": [1, True]}, questions)
    assert structured.state == '{"label":"é","items":[1,true]}'
    assert not structured.truncated
    assert structured.truncation_note == ""
    assert prepare_state("  unchanged\n", questions).state == "  unchanged\n"


@pytest.mark.parametrize(
    "text",
    ["x" * 40_000, "😀" * 20_000, '"\n' * 40_000],
    ids=["ascii", "unicode", "json-escaping"],
)
def test_state_truncation_accounts_for_full_wire_json_and_unicode(text: str):
    questions = {"q": Noul(instructions="Is it relevant?")}
    result = prepare_state(text, questions)
    assert result.truncated
    assert text.startswith(result.state)
    assert request_size(result.state, questions) <= MAX_REQUEST_BYTES
    assert request_size(text[: len(result.state) + 1], questions) > MAX_REQUEST_BYTES
    assert "UTF-8" in result.truncation_note
    assert "token" in result.truncation_note


def test_questions_reduce_the_budget_available_to_state():
    short = {"q": Noul(instructions="Short?")}
    long = {"q": Noul(instructions="Long?" * 1_000)}
    assert len(prepare_state("x" * 40_000, long).state) < len(
        prepare_state("x" * 40_000, short).state
    )


def test_oversized_question_or_criteria_is_rejected_without_truncation():
    questions = {
        "q": Choice(instructions="Pick", criteria={"a": "x" * 33_000, "b": "B"})
    }
    with pytest.raises(ValueError, match="questions.*budget"):
        prepare_state("state", questions)
    assert questions["q"].criteria["a"] == "x" * 33_000


def test_no_questions_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        prepare_state("state", {})


@pytest.mark.parametrize("state", [object(), float("nan"), {"bad": float("inf")}])
def test_non_json_state_is_rejected(state: object):
    with pytest.raises(ValueError, match="JSON"):
        prepare_state(state, {"q": Noul(instructions="Is it relevant?")})


def test_truncated_structured_state_is_an_explicit_text_prefix():
    original = {"content": "a" * 40_000}
    result = prepare_state(original, {"q": Noul(instructions="Relevant?")})
    assert result.truncated
    assert json.dumps(original, separators=(",", ":")).startswith(result.state)
    assert "JSON text" in result.truncation_note


@pytest.mark.parametrize(
    "configured_model", ["", "   ", "x", "  x  ", "jev-custom-" + "x" * 100]
)
def test_boundary_helper_uses_configured_model(monkeypatch, configured_model):
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", configured_model)
    questions = {"q": Noul(instructions="Relevant?")}
    text = "x" * 40_000
    result = prepare_state(text, questions)
    assert request_size(result.state, questions) <= MAX_REQUEST_BYTES
    assert request_size(text[: len(result.state) + 1], questions) > MAX_REQUEST_BYTES
