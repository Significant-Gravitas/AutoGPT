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

from .delegate_to_expert import _handoff_message
from .handoff_to_expert import _transfer_message

_CASES = json.loads(
    (pathlib.Path(__file__).parent / "delegation_preamble_cases.json").read_text()
)["cases"]

_BUILDERS = {"delegation": _handoff_message, "handoff": _transfer_message}


@pytest.mark.parametrize("case", _CASES, ids=[c["kind"] for c in _CASES])
def test_preamble_matches_the_shared_frontend_contract(case: dict[str, str]):
    build = _BUILDERS[case["kind"]]
    assert build(case["caller"], case["context"], case["prompt"]) == case["message"]


def test_every_builder_has_a_case():
    assert {c["kind"] for c in _CASES} == set(_BUILDERS)
