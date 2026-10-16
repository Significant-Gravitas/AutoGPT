"""Behaviour fingerprints: formatting-insensitive, negation-sensitive."""

from __future__ import annotations

from .fingerprint import (
    UNCERTAIN_EQUIVALENCE_THRESHOLD,
    behavior_fingerprint,
    behavior_tokens,
    token_overlap,
)

BASE = "## Steps\n1. Open the file with utf-8 encoding\n2. Validate the row count\n"


def test_reordered_and_reformatted_steps_share_a_fingerprint() -> None:
    reordered = (
        "## Steps\n- Validate the row count\n-   Open the file with UTF-8 encoding\n"
    )
    assert behavior_fingerprint("csv", BASE) == behavior_fingerprint("csv", reordered)


def test_negation_and_thresholds_change_the_fingerprint() -> None:
    negated = BASE.replace("Validate", "Do not validate")
    assert behavior_fingerprint("csv", BASE) != behavior_fingerprint("csv", negated)
    assert "not" in behavior_tokens(negated)
    threshold_a = "## Steps\n1. Retry at most 3 times\n"
    threshold_b = "## Steps\n1. Retry at most 30 times\n"
    assert behavior_fingerprint("x", threshold_a) != behavior_fingerprint(
        "x", threshold_b
    )


def test_light_paraphrase_is_uncertain_not_equal() -> None:
    paraphrase = "## Steps\n1. Open the file using utf-8 encoding first\n2. Validate the row count afterwards\n"
    assert behavior_fingerprint("csv", BASE) != behavior_fingerprint("csv", paraphrase)
    assert (
        token_overlap(behavior_tokens(BASE), behavior_tokens(paraphrase))
        >= UNCERTAIN_EQUIVALENCE_THRESHOLD
    )


def test_unrelated_procedures_do_not_overlap() -> None:
    other = "## Steps\n1. Deploy the container\n2. Tail the logs\n"
    assert token_overlap(behavior_tokens(BASE), behavior_tokens(other)) < 0.2


def test_single_digit_limits_remain_distinct() -> None:
    assert behavior_fingerprint("retry", "1. Retry at most 3 times") != (
        behavior_fingerprint("retry", "1. Retry at most 4 times")
    )
    assert "3" in behavior_tokens("1. Retry at most 3 times")


def test_empty_bullets_do_not_hide_a_prose_procedure() -> None:
    prose = "Validate record counts before importing."
    padded = "* " + " " * 100_000 + "\n" + prose
    assert behavior_tokens(padded) == behavior_tokens(prose)
