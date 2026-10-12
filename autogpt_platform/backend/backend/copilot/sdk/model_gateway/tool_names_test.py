import re

import pytest

from backend.copilot.sdk.model_gateway.tool_names import (
    cli_tool_id,
    name_map,
    upstream_call_id,
    upstream_tool_name,
)

_VALID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@pytest.mark.parametrize(
    "name", ["mcp__copilot__memory_search", "Bash", "read-doc_2", "x" * 64]
)
def test_valid_names_pass_unchanged(name):
    assert upstream_tool_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "mcp__github__repos.get",
        "mcp__srv__" + "very_long_tool_name_" * 5,
        "tool with spaces",
        "ünïcode",
    ],
)
def test_other_names_get_a_valid_stable_stand_in(name):
    stand_in = upstream_tool_name(name)
    assert _VALID.fullmatch(stand_in)
    assert stand_in == upstream_tool_name(name)
    assert name_map([name])[stand_in] == name


def test_two_long_names_with_a_shared_prefix_do_not_collide():
    first = "mcp__server__" + "a" * 80 + "_one"
    second = "mcp__server__" + "a" * 80 + "_two"
    assert upstream_tool_name(first) != upstream_tool_name(second)


def test_call_ids_are_kept_when_valid_and_replaced_stably_when_not():
    assert upstream_call_id("toolu_01ABCdef") == "toolu_01ABCdef"
    long_id = "call_" + "z" * 100
    assert _VALID.fullmatch(upstream_call_id(long_id))
    assert upstream_call_id(long_id) == upstream_call_id(long_id)
    assert upstream_call_id("call.with:dots") != "call.with:dots"


def test_upstream_ids_reach_the_cli_in_anthropic_form():
    assert cli_tool_id("call_abc123") == "call_abc123"
    assert _VALID.fullmatch(cli_tool_id("chatcmpl-tool-9f8e.7d6c"))
    generated = cli_tool_id(None)
    assert generated.startswith("toolu_") and generated != cli_tool_id("")
