from collections import Counter

from .examples import Verdict, load_examples


def test_the_fifty_examples_load_with_known_verdicts():
    examples = load_examples()

    assert len(examples) == 50
    assert [e.id for e in examples] == list(range(1, 51))


def test_the_mix_matches_the_rubric_page():
    mix = Counter(e.verdict for e in load_examples())

    assert mix == {
        Verdict.TASK: 20,
        Verdict.PARENT_WITH_SUBTASKS: 5,
        Verdict.NOT_A_TASK: 12,
        Verdict.UPDATE_EXISTING: 6,
        Verdict.ROUTINE: 5,
        Verdict.ASK_THEN_TASK: 2,
    }


def test_every_hiring_example_is_one_task():
    hires = [e for e in load_examples() if e.input.lower().startswith("hire")]

    assert hires
    assert {e.verdict for e in hires} == {Verdict.TASK}
