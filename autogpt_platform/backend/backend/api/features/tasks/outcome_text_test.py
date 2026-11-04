import pytest

from backend.api.features.tasks.outcome_text import (
    ask_line,
    reads_as_failure,
    result_line,
)


def test_a_result_line_is_the_first_sentence_without_markdown():
    text = "## Pick\n\n**Linear** is the best fit. It is fast.\n\n- cheap\n- simple"

    assert result_line(text) == "Linear is the best fit."


def test_a_long_result_line_is_cut_at_a_word():
    line = result_line("word " * 60)

    assert len(line) <= 140
    assert line.endswith("…")


def test_an_empty_message_has_no_result_line():
    assert result_line(None) == ""
    assert result_line("```\ncode only\n```") == ""


def test_the_ask_is_the_question_the_message_ends_on():
    text = "I found three tools. Which one should I compare first?"

    assert ask_line(text) == "Which one should I compare first?"
    assert ask_line("Connect Slack above.") == "Connect Slack above."


@pytest.mark.parametrize(
    "text",
    [
        "I couldn't find the Q3 report PDF.",
        "No PDF was attached, so there was nothing to summarize.",
        "The billing export wasn't found.",
        "Unable to access the shared drive.",
        "The attachment is missing.",
        "Missing the input file.",
    ],
)
def test_a_result_saying_the_work_wasnt_done_reads_as_failure(text):
    assert reads_as_failure(text)


@pytest.mark.parametrize(
    "text",
    [
        "Use Linear: it fits a team of 8.",
        "Checked the copy and couldn't find any typos.",
        "Fixed the failed login test.",
        "Summarized the report in three bullets.",
    ],
)
def test_a_real_result_does_not_read_as_failure(text):
    assert not reads_as_failure(text)
