"""The eval case for SECRT-3158: an expert's reply names no tool, file path,
shell command or internal id. Pure checks, so they run without the roster."""

from .models import ScoredResponse
from .plain_language import TOOL_NAMES, technical_terms
from .runner import summarize_expert

_TOOLS = frozenset({"run_capability", "bash_exec", "TodoWrite", "Read", "web_search"})


def test_a_plain_reply_passes():
    reply = (
        "Your launch brief is drafted: launch on 25 November at Rs 899, sold on "
        "Shopify and Instagram. Next, tell me the vitamin C percentage. "
        "Read it over and I'll write the first posts."
    )
    assert technical_terms(reply, _TOOLS) == []


def test_tool_names_are_caught_but_not_the_english_words_they_resemble():
    reply = "I'll call run_capability on the hire_expert tool, then Read the notes."
    assert technical_terms(reply, _TOOLS) == ["run_capability"]
    assert technical_terms("Updated the plan with `tool:list_schedules`.", _TOOLS) == [
        "tool:list_schedules"
    ]


def test_file_paths_and_shell_commands_are_caught():
    reply = (
        "Patched ~/workspace/dewleaf/brief.md with sed -i 's/TBD/25 Nov/' and "
        "checked calendar.csv."
    )
    assert technical_terms(reply, _TOOLS) == [
        "~/workspace/dewleaf/brief.md",
        "calendar.csv",
        "sed -i",
    ]


def test_ids_are_caught():
    reply = "Saved as 2616d1f1-16bb-43c0-9f0e-6a7c2a9d0b11."
    assert technical_terms(reply, _TOOLS) == ["2616d1f1-16bb-43c0-9f0e-6a7c2a9d0b11"]


def test_a_shared_file_link_is_not_a_leak():
    reply = "Here's the calendar: [calendar.csv](workspace://abc123#text/csv)."
    assert technical_terms(reply, _TOOLS) == []


def test_the_registry_names_are_the_ones_checked():
    assert "bash_exec" in TOOL_NAMES and "run_capability" in TOOL_NAMES


def test_the_summary_counts_replies_that_leak():
    rows = [
        _row("Maya", "Your brief is ready.", []),
        _row("Maya", "Ran sed -i on brief.md", ["brief.md", "sed -i"]),
        _row("Maya", "Called bash_exec", ["bash_exec"]),
    ]
    assert summarize_expert("Maya", rows).technical_replies == 2


def _row(expert: str, response: str, terms: list[str]) -> ScoredResponse:
    return ScoredResponse(
        expert=expert,
        spec_expert=expert,
        arm="expert",
        kind="briefing",
        prompt_id=f"{expert}-{len(response)}",
        repeat=0,
        model="m",
        response=response,
        technical_terms=terms,
        score=80.0,
    )
