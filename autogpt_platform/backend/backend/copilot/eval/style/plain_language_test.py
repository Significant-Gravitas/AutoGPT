"""The eval case for SECRT-3158: an expert's reply names no tool, file path,
shell command or internal id. Pure checks, so they run without the roster."""

from unittest.mock import AsyncMock, patch

import pytest

from .models import ScoredResponse, StyleEvalResult
from .plain_language import TOOL_NAMES, leaking_experts, technical_terms
from .runner import main, summarize_expert

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


def test_a_link_label_is_still_read():
    reply = "Done with [bash_exec](https://example.com/docs)."
    assert technical_terms(reply, _TOOLS) == ["bash_exec"]


def test_paths_in_inline_code_are_caught():
    assert technical_terms("Saved it to `~/workspace/drafts`.", _TOOLS) == [
        "~/workspace/drafts"
    ]


def test_common_subcommands_are_caught():
    reply = "I ran git status, then npm install."
    assert technical_terms(reply, _TOOLS) == ["git status", "npm install"]


def test_a_command_after_a_run_verb_is_caught():
    reply = "I'm running cat brief now, then executing python3 report."
    assert technical_terms(reply, _TOOLS) == ["cat brief", "python3 report"]
    assert technical_terms("cat brief shows the dates.", _TOOLS) == ["cat brief"]


def test_a_command_in_inline_code_after_a_run_verb_is_caught():
    assert technical_terms("I ran `cat brief` to check.", _TOOLS) == ["cat brief"]
    assert technical_terms("Running 'python3 report' now.", _TOOLS) == [
        "python3 report"
    ]


def test_ordinary_words_after_a_run_verb_stay_plain():
    reply = "Running the cat food promo this week. Ran the head office rota."
    assert technical_terms(reply, _TOOLS) == []
    reply = "Running sort on items now. Running export of orders. Ran find for dupes."
    assert technical_terms(reply, _TOOLS) == []


def test_a_command_that_is_also_a_word_still_counts_with_an_argument():
    reply = "Running sort orders now. Running make build next."
    assert technical_terms(reply, _TOOLS) == ["sort orders", "make build"]


def test_a_product_name_ending_in_js_is_not_a_file():
    assert technical_terms("Your store runs on Next.js and Node.js.", _TOOLS) == []
    assert technical_terms("I updated index.js for you.", _TOOLS) == ["index.js"]


def test_the_registry_names_are_the_ones_checked():
    assert "bash_exec" in TOOL_NAMES and "run_capability" in TOOL_NAMES


def test_the_summary_counts_replies_that_leak():
    rows = [
        _row("Maya", "Your brief is ready.", []),
        _row("Maya", "Ran sed -i on brief.md", ["brief.md", "sed -i"]),
        _row("Maya", "Called bash_exec", ["bash_exec"]),
    ]
    assert summarize_expert("Maya", rows).technical_replies == 2


def test_the_gate_names_every_expert_whose_replies_leak():
    summaries = [
        summarize_expert("Maya", [_row("Maya", "Ran sed -i", ["sed -i"])]),
        summarize_expert("Max", [_row("Max", "Your brief is ready.", [])]),
    ]
    assert leaking_experts(summaries) == ["Maya"]


@pytest.mark.parametrize("terms, exits", [(["sed -i"], True), ([], False)])
def test_a_run_whose_replies_leak_fails_the_command(terms, exits):
    result = _result([_row("Maya", "reply", terms)])
    with (
        patch("sys.argv", ["expert-style-eval"]),
        patch("backend.copilot.eval.style.runner.run", AsyncMock(return_value=result)),
    ):
        if exits:
            with pytest.raises(SystemExit) as raised:
                main()
            assert raised.value.code == 1
        else:
            main()


def _result(rows: list[ScoredResponse]) -> StyleEvalResult:
    return StyleEvalResult(
        run_id="r",
        ts="t",
        fingerprint="f",
        chat_model="m",
        judge_model="j",
        experts=[summarize_expert("Maya", rows)],
        separation=None,
        comparison=[],
        cost_usd=0.0,
        cost_known=True,
        input_tokens=0,
        output_tokens=0,
        responses=rows,
    )


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
