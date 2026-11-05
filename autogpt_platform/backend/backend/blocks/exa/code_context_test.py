"""Tests for ExaCodeContextBlock response parsing."""

import pytest

from backend.blocks.exa.code_context import CodeContextResponse, cost_dollars_text


@pytest.mark.parametrize(
    "raw, expected",
    [("0.005", "0.005"), ({"total": 0.01}, "0.01"), (None, ""), ({}, "")],
)
def test_cost_dollars_text_accepts_string_and_object(raw, expected):
    assert cost_dollars_text(raw) == expected


def test_from_api_tolerates_object_cost():
    parsed = CodeContextResponse.from_api({"costDollars": {"total": 0.02}})
    assert parsed.cost_dollars == "0.02"
