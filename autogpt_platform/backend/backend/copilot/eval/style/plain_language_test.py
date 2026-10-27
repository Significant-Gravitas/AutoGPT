"""The eval case for SECRT-3158: an expert's reply names no tool, file path,
shell command or internal id. Pure checks, so they run without the roster."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from .models import ScoredResponse, StyleEvalResult
from .plain_language import (
    RULES_PATH,
    TOOL_NAMES,
    leaking_experts,
    possible_technical_terms,
    technical_terms,
)
from .runner import main, summarize_expert

_TOOLS = frozenset({"run_capability", "bash_exec", "TodoWrite", "Read", "web_search"})


@pytest.mark.parametrize(
    "example",
    json.loads(RULES_PATH.read_text())["examples"],
    ids=lambda example: example["text"][:40],
)
def test_the_shared_examples_hold_here_as_in_the_chat(example):
    """The chat's plainText.test.ts runs the same examples: one rule set."""
    assert technical_terms(example["text"], ()) == example["certain"]
    assert possible_technical_terms(example["text"]) == example["possible"]


def test_a_plain_reply_passes():
    reply = (
        "Your launch brief is drafted: launch on 25 November at Rs 899, sold on "
        "Shopify and Instagram. Next, tell me the vitamin C percentage. "
        "Read it over and I'll write the first posts."
    )
    assert technical_terms(reply, _TOOLS) == []


def test_tool_names_are_caught_but_not_the_english_words_they_resemble():
    reply = "I'll call run_capability on the hire_expert tool, then Read the notes."
    assert technical_terms(reply, _TOOLS) == ["run_capability", "hire_expert"]
    assert technical_terms("Updated the plan with `tool:list_schedules`.", _TOOLS) == [
        "tool:list_schedules"
    ]


def test_sdk_tool_names_that_are_not_words_are_caught():
    assert technical_terms("I used Glob to find the drafts.", _TOOLS) == ["Glob"]
    assert technical_terms("Grep found two TBD lines.", _TOOLS) == ["Grep"]


def test_sdk_tool_names_that_are_words_are_caught_only_as_tools():
    for reply, terms in [
        ("I opened it with the Read tool.", ["Read"]),
        ("Then `Edit` fixed the date.", ["Edit"]),
        ("I handed it to the Task tool and a Bash tool.", ["Task", "Bash"]),
        ("Read the brief, then Edit the post. Task list below.", []),
        ("Write me a caption; it's a big bash on Friday.", []),
        ("A glob of serum goes a long way.", []),
    ]:
        assert technical_terms(reply, _TOOLS) == terms


def test_single_segment_paths_are_caught():
    reply = "I put the copy in /tmp and the notes in ~/drafts, then ./out."
    assert technical_terms(reply, _TOOLS) == ["/tmp", "~/drafts", "./out"]
    assert technical_terms("Look in `/workspace` for it.", _TOOLS) == ["/workspace"]


def test_slashes_in_prose_are_not_paths():
    reply = "Open 24/7, and/or by appointment; sizes S/M/L at Rs 899 / 30ml."
    assert technical_terms(reply, _TOOLS) == []


def test_plain_file_names_are_caught():
    reply = "I saved report.pdf, pitch-deck.pptx and logo.png for you."
    assert technical_terms(reply, _TOOLS) == [
        "report.pdf",
        "pitch-deck.pptx",
        "logo.png",
    ]
    assert technical_terms("Also budget.docx and orders.sql.", _TOOLS) == [
        "budget.docx",
        "orders.sql",
    ]


def test_document_kinds_in_prose_are_not_file_names():
    reply = (
        "Here is the PDF report and a PNG logo, e.g. for Instagram. "
        "Shop at dewleaf.in, 9 a.m. to 6 p.m."
    )
    assert technical_terms(reply, _TOOLS) == []


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
    reply = "I'm running grep notes now, then executing python3 report."
    assert technical_terms(reply, _TOOLS) == ["grep notes", "python3 report"]
    assert technical_terms("jq orders shows the dates.", _TOOLS) == ["jq orders"]


def test_a_bare_word_like_command_is_reported_but_never_fails_the_gate():
    """ "Running cat brief" reads like a command; "ran head office numbers" is
    prose. Bare, the two can't be told apart, so the gate stays on what is
    certain and the eval reports the rest (SECRT-3158)."""
    for reply, possible in [
        ("I'm running cat brief now.", ["cat brief"]),
        ("cat brief shows the dates.", ["cat brief"]),
        ("I ran head office numbers for you.", ["head office"]),
        ("Running export reports next.", ["export reports"]),
    ]:
        assert technical_terms(reply, _TOOLS) == []
        assert possible_technical_terms(reply) == possible


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


def test_a_command_that_is_also_a_word_is_possible_with_an_argument():
    reply = "Running sort orders now. Running make build next."
    assert technical_terms(reply, _TOOLS) == []
    assert possible_technical_terms(reply) == ["sort orders", "make build"]
    assert possible_technical_terms("Running sort on items now.") == []


def test_a_product_name_ending_in_js_is_not_a_file():
    assert technical_terms("Your store runs on Next.js and Node.js.", _TOOLS) == []
    assert technical_terms("I updated index.js for you.", _TOOLS) == ["index.js"]


@pytest.mark.parametrize(
    "reply, terms",
    [
        ("Running Bash script now.", ["Bash script"]),
        ("I Ran Git Status, then NPM Install.", ["Git Status", "NPM Install"]),
        ("Running Sed -i on the brief.", ["Sed -i"]),
        ("Calling Bash_Exec next.", ["Bash_Exec"]),
        ("Executing Python3 report.", ["Python3 report"]),
        ("Saved Brief.MD and Calendar.CSV.", ["Brief.MD", "Calendar.CSV"]),
        ("Saved Index.JS.", ["Index.JS"]),
        ("I ran `Cat brief`.", ["Cat brief"]),
        ("Saved it to ~/Workspace/Drafts and /TMP.", ["~/Workspace/Drafts", "/TMP"]),
    ],
)
def test_technical_terms_are_caught_whatever_the_case(reply, terms):
    assert technical_terms(reply, _TOOLS) == terms


@pytest.mark.parametrize(
    "reply",
    [
        "Running the CD sales promo this week.",
        "Ran the RM spend report.",
        "Your store runs on Next.JS.",
        "Running the Cat Food promo.",
    ],
)
def test_capitalised_prose_stays_plain(reply):
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


def test_the_summary_counts_possible_leaks_apart_from_the_gate():
    row = _row("Maya", "Running cat brief now.", [])
    row.possible_terms = ["cat brief"]
    summary = summarize_expert("Maya", [row])
    assert summary.possible_technical_replies == 1
    assert summary.technical_replies == 0
    assert leaking_experts([summary]) == []


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
