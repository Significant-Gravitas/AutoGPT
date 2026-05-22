"""Tests for the skill registry (frontmatter parsing + rendering + the
``<available_skills>`` index builder)."""

import pytest

from backend.copilot.tools.skills import (
    DEFAULT_SKILLS,
    MAX_DESCRIPTION_CHARS,
    MAX_NAME_CHARS,
    ParsedSkill,
    _validate_name,
    get_default_skills,
    parse_skill_markdown,
    render_skill_markdown,
    render_skills_index,
)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_render_then_parse_is_identity_for_required_fields():
    """A skill rendered to SKILL.md and parsed back must preserve name,
    description, body, and triggers — anything else is a bug in the
    on-the-wire format helpers."""
    original = ParsedSkill(
        name="example_skill",
        description="An example for the round-trip test.",
        body="## Steps\n1. Do the thing\n2. Done\n",
        triggers=("create_agent", "edit_agent"),
    )
    parsed = parse_skill_markdown(render_skill_markdown(original))
    assert parsed is not None
    assert parsed.name == original.name
    assert parsed.description == original.description
    assert parsed.body.strip() == original.body.strip()
    assert parsed.triggers == original.triggers


def test_parse_rejects_missing_frontmatter():
    """A markdown blob with no YAML frontmatter is not a skill — the
    registry must skip it instead of crashing."""
    assert parse_skill_markdown("just a plain body") is None


def test_parse_rejects_missing_required_fields():
    raw = "---\nname: foo\n---\nbody"
    # No description → invalid skill.
    assert parse_skill_markdown(raw) is None
    raw2 = "---\ndescription: bar\n---\nbody"
    # No name → invalid (and the fallback only kicks in when explicitly passed).
    assert parse_skill_markdown(raw2) is None


def test_parse_accepts_string_triggers_as_csv():
    """Tolerate ``triggers: a, b, c`` strings as a convenience for
    hand-edited SKILL.md files — the canonical render uses a YAML list
    but ingesting both shapes makes the parser forgiving."""
    raw = "---\nname: foo\ndescription: bar\ntriggers: a, b, c\n---\nbody\n"
    parsed = parse_skill_markdown(raw)
    assert parsed is not None
    assert parsed.triggers == ("a", "b", "c")


def test_parse_fallback_name_used_when_frontmatter_missing_name():
    """``list_user_skills`` derives the slug from the folder name. The
    parser must honour that fallback so a SKILL.md authored without an
    explicit ``name`` still surfaces under its folder slug."""
    raw = "---\ndescription: hello\n---\nbody\n"
    parsed = parse_skill_markdown(raw, fallback_name="slug-from-folder")
    assert parsed is not None
    assert parsed.name == "slug-from-folder"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_name_rejects_reserved_default_names():
    """Built-in defaults must not be overwritable via store_skill — they
    ship from disk and have semantics other code (the agent-building
    gate) relies on."""
    for default in DEFAULT_SKILLS:
        assert _validate_name(default.name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "",
        "UPPER",
        "has space",
        "trailing-",
        "-leading",
        "has/slash",
        "has.dot",
        "name_" + "x" * (MAX_NAME_CHARS + 1),
    ],
)
def test_validate_name_rejects_invalid_slugs(name: str):
    assert _validate_name(name) is not None


@pytest.mark.parametrize(
    "name",
    ["foo", "foo_bar", "foo-bar", "a1b2", "snake_case_long_skill_name"],
)
def test_validate_name_accepts_valid_slugs(name: str):
    assert _validate_name(name) is None


# ---------------------------------------------------------------------------
# Defaults + index
# ---------------------------------------------------------------------------


def test_default_skills_load_from_disk():
    """The migration from get_agent_building_guide / get_mcp_guide
    relies on the on-disk markdown files still being present at the
    paths the default registry points to."""
    defaults = get_default_skills()
    names = {s.name for s in defaults}
    assert "agent_building_guide" in names
    assert "mcp_tool_guide" in names
    # Bodies must be non-trivial — a zero-byte file silently kills the
    # whole feature for end users.
    for skill in defaults:
        assert len(skill.body) > 100


def test_render_skills_index_includes_name_and_description():
    skills = [
        ParsedSkill(name="a", description="alpha desc", body=""),
        ParsedSkill(name="b", description="beta desc", body="", triggers=("x", "y")),
    ]
    index = render_skills_index(skills)
    assert "a — alpha desc" in index
    assert "b — beta desc" in index
    assert "triggers: x, y" in index


def test_render_skills_index_empty_returns_empty_string():
    """An empty registry must yield ``""`` so ``inject_user_context``
    omits the ``<available_skills>`` block entirely — preventing a
    zero-content tag from eating tokens for no reason."""
    assert render_skills_index([]) == ""


def test_description_length_cap_enforced_by_validate_path():
    """The cap is enforced inside ``StoreSkillTool._execute`` (not by
    the dataclass), so this test just locks in the constant — bumping
    it requires conscious thought about per-turn token cost."""
    assert MAX_DESCRIPTION_CHARS == 200
