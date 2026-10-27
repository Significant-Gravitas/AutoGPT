import pytest

from backend.copilot.gate.classifier import ACTION_RUBRIC
from backend.copilot.gate.jev import JEV_RUBRIC
from backend.copilot.gate.judged_rubric import (
    _EXAMPLES,
    _SETTLES,
    _SETTLES_JUDGED,
    _SHEETS_EXAMPLE,
    ITEM_3,
    judged,
)


@pytest.mark.parametrize("rubric, style", [(ACTION_RUBRIC, "llm"), (JEV_RUBRIC, "jev")])
def test_only_item_3_its_caveat_and_the_examples_change(rubric, style):
    """The measured rubric is what ships; the variant may differ from it only
    where a judged outward call needs it to."""
    lines = judged(rubric, style).split("\n")
    assert [line for line in lines if line.startswith("3. ")] == [ITEM_3]
    assert not any(_SETTLES in line for line in lines)
    assert any(_SETTLES_JUDGED in line for line in lines)
    assert not any(line.startswith(_SHEETS_EXAMPLE) for line in lines)
    for request, call, word, _ in _EXAMPLES:
        assert any(
            line.startswith(f'- "{request}" / {call} → {word}') for line in lines
        )
    kept = [
        line
        for line in rubric.split("\n")
        if not line.startswith(("3. ", _SHEETS_EXAMPLE)) and _SETTLES not in line
    ]
    assert all(line in lines for line in kept)


def test_a_reworded_sheets_example_refuses_to_build():
    """A reworded copy would slip past the filter and teach the judge that a
    judged write asks."""
    moved = JEV_RUBRIC.replace(_SHEETS_EXAMPLE, '- "Put the totals in row 52."')
    assert moved != JEV_RUBRIC
    with pytest.raises(ValueError):
        judged(moved, "jev")
