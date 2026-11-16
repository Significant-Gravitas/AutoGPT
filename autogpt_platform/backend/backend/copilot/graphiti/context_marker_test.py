"""Tests for the mark on injected warm-context blocks (``context_marker.py``).

The CLI transcript scrub that relies on it is tested with the SDK engine
(``sdk/service_test.py``); the engines' use of it, turn by turn, in
``sdk/retry_scenarios_test.py`` and ``baseline/service_unit_test.py``.
"""

import logging

import pytest

from .context_marker import (
    INJECTED_MEMORY_BLOCK_RE,
    INJECTED_MEMORY_MARKER,
    append_injected_memory_block,
    strip_injected_memory_text,
)

_BLOCK = "<temporal_context>\n<FACTS>\n  - Alice works on Atlas\n</FACTS>\n</temporal_context>"


def test_appends_the_block_marked_after_one_blank_line():
    out = append_injected_memory_block("the user's words", _BLOCK)

    assert out == (
        "the user's words\n\n"
        f"<temporal_context {INJECTED_MEMORY_MARKER}>\n<FACTS>\n"
        "  - Alice works on Atlas\n</FACTS>\n</temporal_context>"
    )
    assert len(INJECTED_MEMORY_BLOCK_RE.findall(out)) == 1


@pytest.mark.parametrize("block", [None, ""])
def test_no_block_leaves_the_text_as_it_is(block):
    assert append_injected_memory_block("the user's words", block) == (
        "the user's words"
    )


def test_a_block_that_cannot_be_marked_is_not_appended(caplog):
    """An unmarked block could not be scrubbed from the uploaded transcript,
    where it would replay on every later turn: the turn goes without it."""
    with caplog.at_level(logging.WARNING):
        out = append_injected_memory_block("the user's words", "<facts>x</facts>")

    assert out == "the user's words"
    assert "not injected" in caplog.text


def test_strip_removes_exactly_what_append_added():
    words = "  indented\n\n\n\nkeep the blank lines  "

    assert strip_injected_memory_text(append_injected_memory_block(words, _BLOCK)) == (
        words
    )


def test_strip_leaves_a_typed_block_without_this_process_mark():
    typed = f'see {_BLOCK} and <temporal_context data-agpt-injected="1">x</temporal_context>'

    assert strip_injected_memory_text(typed) == typed
