"""The text rules of a task's record (pure, no database)."""

import pytest

from backend.data.task_text import (
    PROGRESS_MAX_CHARS,
    SUMMARY_MAX_CHARS,
    clip,
    file_key,
    recorded_tool_names,
    short_result,
    too_long,
    tool_names,
    url_key,
)


def test_a_line_that_fits_is_kept_and_a_long_one_is_cut_at_a_word():
    assert clip("  Compare   three CRMs ", 40) == "Compare three CRMs"
    assert clip("", 40) is None
    cut = clip("Compare the pricing of three CRMs for the sales team", 30)
    assert cut == "Compare the pricing of three…"
    assert cut is not None and len(cut) <= 30


def test_a_line_over_its_cap_is_refused_with_the_cap():
    assert too_long("Reading the pricing pages", PROGRESS_MAX_CHARS, "line") is None
    refusal = too_long("x" * 200, PROGRESS_MAX_CHARS, "progress line")
    assert refusal is not None
    assert "200" in refusal and str(PROGRESS_MAX_CHARS) in refusal


def test_a_short_reply_is_its_own_result():
    assert short_result("Sent the three reminders.") == "Sent the three reminders."
    assert short_result("   ") is None


def test_a_long_report_becomes_its_opening_as_plain_text():
    report = (
        "## Cursor brief\n\n**Pro is $20 a month.** Teams is $40 a seat. "
        "See [pricing](https://cursor.com/pricing).\n\n| Plan | Price |\n|---|---|\n"
        + "More detail follows here. " * 60
    )
    result = short_result(report)
    assert result is not None and len(result) <= SUMMARY_MAX_CHARS
    assert result.startswith("Cursor brief Pro is $20 a month. Teams is $40 a seat.")
    assert not set("#*|`") & set(result)
    assert "https://" not in result
    assert result.endswith(".")


@pytest.mark.parametrize(
    "cited",
    [
        "bash_exec",
        "mcp__copilot__bash_exec",
        "tool:bash_exec",
        "write_file + bash_exec",
        "`bash_exec`, web_fetch",
        "Bash_Exec and web_search",
    ],
)
def test_a_tool_is_found_however_the_worker_writes_its_name(cited):
    assert tool_names(cited) & tool_names("bash_exec")


def test_a_capability_answers_to_its_parts_and_other_tools_do_not_match():
    recorded = recorded_tool_names("mcp:gmail/send_email")
    assert tool_names("send_email") & recorded
    assert tool_names("gmail") & recorded
    assert tool_names("run_capability") & recorded
    assert not tool_names("mcp") & recorded
    assert not tool_names("run_capability") & recorded_tool_names("bash_exec")
    assert not tool_names("stripe_pay") & tool_names("mcp:gmail/send_email")
    assert not tool_names("the tool and then") & tool_names("bash_exec")
    assert tool_names(None) == set()


@pytest.mark.parametrize(
    "named",
    [
        "report.md",
        "REPORT.md",
        "/home/user/report.md",
        "`report.md`",
        "workspace://report.md#text/markdown",
    ],
)
def test_a_file_is_found_by_its_bare_name(named):
    assert file_key(named) == file_key("report.md")


def test_a_page_is_the_same_page_with_or_without_scheme_www_or_slash():
    assert url_key("http://www.cursor.com/pricing/") == url_key(
        "https://cursor.com/pricing"
    )
    assert url_key("https://cursor.com/pricing") != url_key("https://cursor.com/free")
    assert url_key("not a link") == ""
