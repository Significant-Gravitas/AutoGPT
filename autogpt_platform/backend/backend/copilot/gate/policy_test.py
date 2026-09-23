"""The effect table and the per-mode decision — the part no model decides."""

import pytest

from backend.copilot.gate.policy import (
    MCP_FILE_READ_TOOLS,
    MCP_FILE_WRITE_TOOLS,
    Effect,
    Verdict,
    classified_tools,
    effect_for,
    verdict_for,
)
from backend.copilot.tools import TOOL_REGISTRY


def test_every_registry_tool_has_an_effect():
    assert set(TOOL_REGISTRY) - classified_tools() == set()


def test_the_table_names_no_tool_that_does_not_exist():
    stale = classified_tools() - set(TOOL_REGISTRY)
    assert stale == MCP_FILE_READ_TOOLS | MCP_FILE_WRITE_TOOLS


# One tool per effect, each mode's column. Swapping any cell must go red.
@pytest.mark.parametrize(
    "tool, ask_first, auto, unsupervised",
    [
        ("web_search", Verdict.RUN, Verdict.RUN, Verdict.RUN),
        ("write_workspace_file", Verdict.RUN, Verdict.RUN, Verdict.RUN),
        ("Write", Verdict.RUN, Verdict.RUN, Verdict.RUN),
        ("bash_exec", Verdict.ASK, Verdict.JUDGE, Verdict.RUN),
        ("delete_folder", Verdict.ASK, Verdict.JUDGE, Verdict.RUN),
        ("post_to_chat_platform", Verdict.ASK, Verdict.ASK, Verdict.RUN),
        ("run_agent", Verdict.RUN, Verdict.RUN, Verdict.RUN),
    ],
)
def test_each_modes_column(tool, ask_first, auto, unsupervised):
    assert verdict_for("ask_first", tool) is ask_first
    assert verdict_for("auto", tool) is auto
    assert verdict_for("unsupervised", tool) is unsupervised


def test_an_unclassified_tool_is_a_platform_edit():
    assert effect_for("some_new_tool") is Effect.PLATFORM
    assert verdict_for("ask_first", "some_new_tool") is Verdict.ASK
    assert verdict_for("auto", "some_new_tool") is Verdict.JUDGE


@pytest.mark.parametrize(
    "tool",
    ["connect_integration", "request_credential_grant", "resume_capability"],
)
def test_questions_to_the_user_and_completions_are_never_gated(tool):
    assert effect_for(tool) is Effect.UNGATED


@pytest.mark.parametrize("tool", ["run_capability", "run_agent"])
def test_capability_and_workflow_runs_stay_on_the_irreversible_pause(tool):
    """They get a subject the mode can decide on in L5a; until then the
    irreversible-action pause is their gate."""
    assert effect_for(tool) is Effect.UNGATED


def test_retired_tool_names_are_gone():
    retired = {"run_block", "run_mcp_tool", "find_block", "get_mcp_guide"}
    assert retired & classified_tools() == set()


def test_reading_a_truncated_tool_result_is_a_read():
    from backend.copilot.sdk.tool_adapter import _READ_TOOL_NAME

    assert effect_for(_READ_TOOL_NAME) is Effect.READ
