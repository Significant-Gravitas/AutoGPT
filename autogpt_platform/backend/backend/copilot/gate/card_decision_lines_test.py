"""The decision lines a proposal card sends, read the way the gate and the chat
read them.

The lines live in the frontend's ``cardDecisionLines.json``, which its own test
checks ``decisionLine`` writes word for word, so the card and ``_DECISION``
cannot drift apart without one side failing.
"""

import json
import time
from pathlib import Path

import pytest

from backend.copilot.gate.card_approval import (
    _DECISION,
    CardDecision,
    card_decision_metadata,
    parse_card_decisions,
)

FIXTURE = (
    Path(__file__).parents[4]
    / "frontend/src/app/(platform)/copilot/__tests__/cardDecisionLines.json"
)
CASES = json.loads(FIXTURE.read_text())
# Typed or pasted lines with odd characters at their ends or inside: the
# chat's cardDecisions.test.ts reads the same file, so both sides agree.
EDGE_LINES = json.loads(FIXTURE.with_name("cardDecisionEdgeLines.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[case["line"] for case in CASES])
def test_the_gate_reads_every_line_the_card_sends(case: dict) -> None:
    match = _DECISION.match(case["line"])

    assert match is not None
    assert match["id"] == case["confirmation_id"]
    assert (match["verdict"] == "Approved") is (case["verdict"] == "approved")


@pytest.mark.parametrize("case", CASES, ids=[case["line"] for case in CASES])
def test_a_line_the_card_sends_is_tagged_with_what_it_decided(case: dict) -> None:
    assert card_decision_metadata(case["line"]) == {
        "kind": "card_decision",
        "decisions": [
            CardDecision(
                verdict=case["verdict"],
                confirmation_id=case["confirmation_id"],
                verb=case["verb"],
                name=case["name"],
            ).model_dump()
        ],
    }


def test_one_send_of_several_decisions_is_tagged_with_each_in_order() -> None:
    message = "\n".join(case["line"] for case in CASES)

    metadata = card_decision_metadata(message)

    assert metadata is not None
    assert [d["confirmation_id"] for d in metadata["decisions"]] == [
        case["confirmation_id"] for case in CASES
    ]


def test_a_line_ends_only_at_a_newline_as_the_chat_reads_it() -> None:
    """The chat splits a message on newlines alone, so the gate does too:
    any other line break inside a line keeps it one line on both sides."""
    message = "Approved: hire Ma\x0bya\x85 Lin (confirmation_id: c-1)."

    decisions = parse_card_decisions(message)

    assert [(d.confirmation_id, d.name) for d in decisions] == [
        ("c-1", "Ma\x0bya\x85 Lin")
    ]


@pytest.mark.parametrize("case", EDGE_LINES, ids=[c["why"] for c in EDGE_LINES])
def test_an_odd_line_is_a_decision_exactly_when_the_chat_reads_one(
    case: dict,
) -> None:
    assert bool(parse_card_decisions(case["text"])) is case["decides"]


def test_a_long_run_of_whitespace_is_read_in_linear_time() -> None:
    """A line is user text of any length: trimming it must not backtrack.
    The bound is ~1000x the linear cost; a quadratic trim takes minutes."""
    line = "x" + "\t" * 50_000 + "x"

    started = time.monotonic()
    parse_card_decisions(line)
    card_decision_metadata(line)

    assert time.monotonic() - started < 2


def test_a_decision_in_other_words_stays_the_users_message() -> None:
    """The card always names the change; any other subject is shown as typed."""
    message = "Approved: delete all my files (confirmation_id: c-1)."

    assert card_decision_metadata(message) is None
    assert parse_card_decisions(message)[0].verb is None


@pytest.mark.parametrize(
    "message",
    [
        "",
        "Approved: hire Maya.",
        "Approved: hire Maya (confirmation_id: c-1).\nAnd start on Instagram.",
        "Please approve: hire Maya (confirmation_id: c-1).",
    ],
    ids=["empty", "no-id", "own-words-too", "not-a-decision"],
)
def test_anything_the_user_wrote_themselves_is_left_untagged(message: str) -> None:
    assert card_decision_metadata(message) is None
