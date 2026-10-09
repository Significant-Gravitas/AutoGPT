"""Skills as capability entries: what the index sees and how ids read."""

import pytest

from backend.copilot.tools.skills import ParsedSkill

from .skills import SKILL_ID_PREFIX, skill_entries, skill_name

TRIAGE = ParsedSkill(
    name="triage-and-prioritize",
    description="Triage a support ticket and set its priority.",
    body="1. read the ticket\n2. set the priority",
    triggers=("triage", "prioritise ticket"),
)


def test_skill_entry_indexes_description_and_triggers_but_lists_the_purpose():
    (entry,) = skill_entries([TRIAGE])
    assert entry.id == f"{SKILL_ID_PREFIX}triage-and-prioritize"
    assert entry.kind == "skill" and entry.context == "direct"
    assert entry.implementations[0].ref == "triage-and-prioritize"
    assert entry.purpose == TRIAGE.description
    assert entry.description.endswith("Triggers: triage, prioritise ticket.")
    assert "skill" in entry.tags
    listing = entry.listing()
    assert listing["kind"] == "skill" and "description" not in listing
    assert "connected" not in listing


def test_a_skill_without_triggers_indexes_its_description_alone():
    (entry,) = skill_entries(
        [ParsedSkill(name="notes", description="Take notes.", body="")]
    )
    assert entry.description == "Take notes."


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("skill:triage", "triage"),
        ("Skill: Triage ", "triage"),
        ("skill:", None),
        ("tool:read_skill", None),
        ("triage", None),
        ("", None),
    ],
)
def test_skill_name_reads_only_skill_ids(raw: str, expected: str | None):
    assert skill_name(raw) == expected
