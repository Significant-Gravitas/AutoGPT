"""The delegation/handoff preamble is a contract with the frontend.

``ChatMessagesContainer`` hides the generated ``[Delegated task from …]`` /
``[Task handed to you by …]`` framing from user bubbles by matching its
wording. These cases are the one place both sides agree on that wording:
reword a preamble here and this test fails until
``delegation_preamble_cases.json`` is updated, after which the frontend
contract test fails until its matcher follows.
"""

import json
import pathlib
import time

import pytest

from backend.copilot.delegation_preamble import NAME_MAX_LENGTH, strip_teammate_preamble

from .delegate_to_expert import _handoff_message
from .expert_delegation import CALLER_NAME_LIMIT, USER_NAME_LIMIT, first_name
from .handoff_to_expert import _transfer_message

_CASES = json.loads(
    (pathlib.Path(__file__).parent / "delegation_preamble_cases.json").read_text()
)["cases"]

_BUILDERS = {"delegation": _handoff_message, "handoff": _transfer_message}


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[
        f"{c['kind']}-{c['user'] or 'unnamed'}-{'context' if c['context'] else 'bare'}"
        for c in _CASES
    ],
)
def test_preamble_matches_the_shared_frontend_contract(case: dict[str, str]):
    build = _BUILDERS[case["kind"]]
    message = build(
        case["caller"], case["context"], case["prompt"], case["user"] or None
    )
    assert message == case["message"]


def test_every_builder_has_a_case():
    assert {c["kind"] for c in _CASES} == set(_BUILDERS)


def test_every_builder_has_a_case_with_and_without_a_user_name():
    """Both wordings must stay in the contract: the named one is what new
    rows carry, the "this user's" one is every stored row and the fallback."""
    assert {(c["kind"], bool(c["user"])) for c in _CASES} == {
        (kind, named) for kind in _BUILDERS for named in (True, False)
    }


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[
        f"{c['kind']}-{c['user'] or 'unnamed'}-{'context' if c['context'] else 'bare'}"
        for c in _CASES
    ],
)
def test_the_backend_hides_the_preamble_exactly_as_the_frontend_does(
    case: dict[str, str],
):
    """The public share sanitizer strips the preamble on the server, so it
    must agree with the frontend matcher on every shared case."""
    assert strip_teammate_preamble(case["message"]) == case["visible"]


def test_the_matcher_covers_the_longest_names_a_preamble_can_hold():
    assert NAME_MAX_LENGTH >= max(CALLER_NAME_LIMIT, USER_NAME_LIMIT)
    # As the tools build it: the caller capped by safe_caller_name inside the
    # builder, the user by first_name before it.
    row = _handoff_message("A" * 200, "", "Task", first_name("B" * 200))
    assert strip_teammate_preamble(row) == "Task"


def test_a_crafted_line_cannot_stall_the_matcher():
    """Unbounded name classes backtrack quadratically on one long line of
    repeated framing; this runs on the event loop for shares and titles."""
    crafted = "[Delegated task from " + "x, a teammate on y's team " * 2500
    started = time.perf_counter()
    assert strip_teammate_preamble(crafted) == crafted
    # Unbounded: ~0.5s on this input; bounded: well under a millisecond.
    assert time.perf_counter() - started < 0.05
