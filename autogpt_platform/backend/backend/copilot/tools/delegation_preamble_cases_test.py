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

import pytest

from backend.copilot.delegation_preamble import strip_teammate_preamble

from .delegate_to_expert import _handoff_message
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
