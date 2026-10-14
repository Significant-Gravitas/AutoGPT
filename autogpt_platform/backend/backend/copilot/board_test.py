import time

import pytest

from backend.copilot.board import (
    DEFAULT_COLUMNS,
    LAST_LINE_MAX,
    MAX_BOARD_NAME,
    BoardColumn,
    BoardColumnsError,
    BoardNameError,
    first_sentence,
    new_column_key,
    validate_board_name,
    validate_columns,
)


def _cols(*extra: BoardColumn) -> list[BoardColumn]:
    return [*DEFAULT_COLUMNS, *extra]


def test_defaults_are_three_builtins():
    assert [(c.key, c.name, c.builtin) for c in DEFAULT_COLUMNS] == [
        ("backlog", "Backlog", True),
        ("in_progress", "In progress", True),
        ("done", "Done", True),
    ]


def test_validate_strips_names_and_derives_builtin():
    key = new_column_key()
    result = validate_columns(
        [
            BoardColumn(key="backlog", name="  Ideas  ", builtin=False),
            BoardColumn(key="in_progress", name="Doing"),
            BoardColumn(key="done", name="Done"),
            BoardColumn(key=key, name="Review", builtin=True),
        ]
    )
    assert result[0].name == "Ideas" and result[0].builtin is True
    assert result[3].builtin is False


def test_validate_rejects_missing_builtin():
    with pytest.raises(BoardColumnsError):
        validate_columns([c for c in DEFAULT_COLUMNS if c.key != "done"])


def test_validate_rejects_more_than_six():
    extra = [BoardColumn(key=new_column_key(), name=f"C{i}") for i in range(4)]
    with pytest.raises(BoardColumnsError):
        validate_columns(_cols(*extra))


def test_validate_rejects_blank_name_duplicate_key_and_bad_custom_key():
    with pytest.raises(BoardColumnsError):
        validate_columns(_cols(BoardColumn(key=new_column_key(), name="   ")))
    key = new_column_key()
    with pytest.raises(BoardColumnsError):
        validate_columns(
            _cols(BoardColumn(key=key, name="A"), BoardColumn(key=key, name="B"))
        )
    with pytest.raises(BoardColumnsError):
        validate_columns(_cols(BoardColumn(key="review", name="Review")))


def test_validate_rejects_a_custom_key_with_a_trailing_newline():
    with pytest.raises(BoardColumnsError):
        validate_columns(_cols(BoardColumn(key=f"{new_column_key()}\n", name="A")))


def test_validate_rejects_name_over_40_chars():
    with pytest.raises(BoardColumnsError):
        validate_columns(_cols(BoardColumn(key=new_column_key(), name="x" * 41)))


@pytest.mark.parametrize(
    "text,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("Drafted the PRD. Waiting on Notion access.", "Drafted the PRD."),
        ("## Status\n\nAll **done** here! Next up: tests.", "Status All done here!"),
        ("No punctuation at all", "No punctuation at all"),
        ("Version 1.2 is out. Rest.", "Version 1.2 is out."),
        ("1. Drafted the PRD\n2. Sent it", "Drafted the PRD Sent it"),
        ("- Drafted the PRD\n- Sent it. Waiting.", "Drafted the PRD Sent it."),
        ("  3) Booked the room", "Booked the room"),
    ],
)
def test_first_sentence(text, expected):
    assert first_sentence(text) == expected


def test_first_sentence_clips_long_text():
    line = first_sentence("word " * 100)
    assert line is not None and len(line) <= LAST_LINE_MAX and line.endswith("…")


@pytest.mark.parametrize(
    "text,expected",
    [("\n" * 100_000 + "x", "x"), (" \n" * 100_000 + "1. x", "x")],
)
def test_first_sentence_stays_fast_on_long_blank_runs(text, expected):
    started = time.perf_counter()
    line = first_sentence(text)
    assert time.perf_counter() - started < 0.5
    assert line == expected


def test_first_sentence_reads_only_the_start_of_a_long_reply():
    started = time.perf_counter()
    line = first_sentence("- " + "a" * 500_000)
    assert time.perf_counter() - started < 0.5
    assert line is not None and len(line) <= LAST_LINE_MAX


def test_board_names_are_trimmed():
    assert validate_board_name("  Launch  ") == "Launch"


@pytest.mark.parametrize("name", ["", "   ", "x" * (MAX_BOARD_NAME + 1)])
def test_blank_or_long_board_names_are_refused(name):
    with pytest.raises(
        BoardNameError, match="Project names must be 1 to 40 characters"
    ):
        validate_board_name(name)


def test_a_board_name_at_the_limit_is_kept():
    assert validate_board_name("x" * MAX_BOARD_NAME) == "x" * MAX_BOARD_NAME
