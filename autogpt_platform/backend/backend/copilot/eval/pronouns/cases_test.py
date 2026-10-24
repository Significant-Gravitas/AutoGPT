"""The pronoun eval's cases and check, with no model in the loop."""

import pytest

from backend.api.features.experts.models import USER_ADDRESS_RULE
from backend.copilot.expert_context import render_expert_identity_suffix

from .cases import (
    PronounCase,
    expand_cases,
    first_message,
    gendered_pronouns,
    load_case_file,
    teammate_expert,
)

CASE_FILE = load_case_file()
CASES = expand_cases(CASE_FILE)
TEAMMATE = teammate_expert(CASE_FILE.teammate)


def _case(persona_id: str, kind: str) -> PronounCase:
    return next(c for c in CASES if c.persona.id == persona_id and c.kind == kind)


def test_every_persona_meets_every_brief_and_task():
    per_persona = len(CASE_FILE.otto_briefs) + len(CASE_FILE.expert_tasks)
    assert len(CASES) == len(CASE_FILE.personas) * per_persona
    assert len({c.id for c in CASES}) == len(CASES)
    assert any(p.user_name is None for p in CASE_FILE.personas)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Can he send a photo of the serum?", ["he"]),
        ("so he wants you to run the launch planning for him", ["he", "him"]),
        (
            "Ask Priya for her logo. His team? She'd know. Herself.",
            ["her", "His", "She", "Herself"],
        ),
        ("Can you send a photo? Priya, can they email the waitlist?", []),
        ("The theme is here, there and everywhere; shed the hero's helmet.", []),
    ],
)
def test_the_check_finds_gendered_pronouns_and_only_those(text, expected):
    assert gendered_pronouns(text) == expected


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_no_case_primes_a_gender(case: PronounCase):
    """The input itself must not hand the model a pronoun to copy."""
    assert gendered_pronouns(first_message(case, TEAMMATE)) == []


def test_otto_reads_the_owners_profile_and_the_teammate():
    message = first_message(_case("priya-dewleaf", "otto_brief"), TEAMMATE)
    assert "<user_context>" in message and "Name: Priya" in message
    assert "<team_context>" in message and "Maya" in message


@pytest.mark.parametrize(
    ("persona_id", "team"),
    [
        ("priya-dewleaf", "a teammate on Priya's team"),
        ("sam-harbor-bikes", "a teammate on Sam's team"),
        ("unnamed-kiln-and-co", "a teammate on this user's team"),
    ],
)
def test_the_expert_gets_the_production_preamble(persona_id, team):
    message = first_message(_case(persona_id, "expert_task"), TEAMMATE)
    assert "[Delegated task from Otto, " + team in message


def test_the_teammate_carries_the_protected_rule():
    assert USER_ADDRESS_RULE in render_expert_identity_suffix(TEAMMATE)
