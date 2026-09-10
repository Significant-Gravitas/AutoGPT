"""The skills a roster expert ships with must stay loadable and in sync.

These are the checks that stop a shipped expert quietly losing its
capability: a file that no longer parses, or a folder that no longer
matches any roster entry after a rename.
"""

import pytest

from backend.api.features.experts import roster_skills
from backend.api.features.experts.seed import ROSTER
from backend.copilot.tools.skills import _NAME_RE, MAX_BODY_CHARS, MAX_DESCRIPTION_CHARS


def _roster_names() -> set[str]:
    return {entry["name"].strip().lower() for entry in ROSTER}


def test_every_skill_folder_matches_a_roster_entry():
    """A renamed roster entry orphans its folder, and the expert then hires
    with no skills and no error. Nothing else would catch that."""
    orphans = sorted(set(roster_skills.template_names_with_skills()) - _roster_names())
    assert not orphans, (
        f"roster_skills folders with no matching roster entry: {orphans}. "
        f"Rename the folder to match, or drop it."
    )


@pytest.mark.parametrize("template", sorted(roster_skills.template_names_with_skills()))
def test_shipped_skills_parse(template: str):
    """Every shipped file must survive the same parser the upload endpoint
    uses, or the hire silently installs fewer skills than the roster claims."""
    folder = roster_skills.CONTENT_DIR / template
    files = sorted(folder.glob("*/SKILL.md"))
    assert files, f"{template} has a skills folder but no SKILL.md in it"

    parsed = roster_skills.skills_for_template(template)
    assert len(parsed) == len(
        files
    ), f"{template}: {len(files)} files but only {len(parsed)} parsed"


@pytest.mark.parametrize("template", sorted(roster_skills.template_names_with_skills()))
def test_shipped_skills_are_installable(template: str):
    """Same limits `store_user_skill` enforces. A skill that violates one
    fails at hire time, per-user, which is a much worse place to find out."""
    for skill in roster_skills.skills_for_template(template):
        assert _NAME_RE.match(skill.name), f"{skill.name!r} is not a valid slug"
        assert skill.description, f"{skill.name} has no description"
        assert (
            len(skill.description) <= MAX_DESCRIPTION_CHARS
        ), f"{skill.name} description is {len(skill.description)} chars"
        assert (
            len(skill.body) <= MAX_BODY_CHARS
        ), f"{skill.name} body is {len(skill.body)} chars"


@pytest.mark.parametrize("template", sorted(roster_skills.template_names_with_skills()))
def test_skill_name_matches_its_folder(template: str):
    """The folder is what the loader globs and the name is what the expert
    ends up owning; if they disagree the two are impossible to match up by
    hand later."""
    folder = roster_skills.CONTENT_DIR / template
    for path in sorted(folder.glob("*/SKILL.md")):
        parsed = roster_skills.skills_for_template(template)
        names = {s.name for s in parsed}
        assert (
            path.parent.name in names
        ), f"{template}/{path.parent.name}: frontmatter name is not the folder name"


def test_unknown_template_ships_nothing():
    """Most roster entries ship no skills; that must be silent, not an error."""
    assert roster_skills.skills_for_template("no-such-expert") == []


def test_lookup_is_case_insensitive():
    """`hire_expert` passes the roster's display name ("Ada"), the folder is
    lowercase."""
    for template in roster_skills.template_names_with_skills():
        assert roster_skills.skills_for_template(template.upper())
