"""A SKILL.md merges frontmatter by field and body by line, so an owner's
edit (which re-renders the frontmatter) never masks a catalog change to a
field, and a field nobody touched keeps the catalog's bytes."""

import yaml

from backend.copilot.tools.skill_merge import merge_skill_markdown
from backend.copilot.tools.skills import ParsedSkill, render_skill_markdown

CATALOG_V1 = "---\nname: demo\ndescription: First description\nversion: 1\n---\n\n# Demo\n\nintro\nkeep\n"
CATALOG_V2 = "---\nname: demo\ndescription: Second description\nversion: 2\n---\n\n# Demo\n\nnew intro\nkeep\n"


def _rerendered(text: str, body: str) -> str:
    """What ``store_skill`` writes when the owner edits the body: the same
    fields, re-rendered by the runtime, so the frontmatter text differs."""
    from backend.copilot.tools.skills import parse_skill_markdown

    parsed = parse_skill_markdown(text)
    assert parsed is not None
    return render_skill_markdown(ParsedSkill(**{**parsed.__dict__, "body": body}))


def test_a_catalog_field_change_reaches_a_body_edited_copy_without_conflict():
    ours = _rerendered(CATALOG_V1, "# Demo\n\nintro\nkeep\nmy addition\n")
    assert ours.splitlines()[1:4] != CATALOG_V1.splitlines()[1:4]  # re-rendered

    merged = merge_skill_markdown(CATALOG_V1, ours, CATALOG_V2)

    assert not merged.conflicted
    front, body = merged.text.split("\n---\n", 1)
    assert yaml.safe_load(front[4:]) == {
        "name": "demo",
        "description": "Second description",
        "version": 2,
    }
    assert "new intro\nkeep\nmy addition" in body


def test_a_field_both_sides_changed_keeps_the_owners_value_and_says_so():
    ours = CATALOG_V1.replace("First description", "My description")

    merged = merge_skill_markdown(CATALOG_V1, ours, CATALOG_V2)

    assert merged.conflicted
    assert "description: My description" in merged.text
    assert "version: 2" in merged.text  # the field only the catalog changed


def test_an_untouched_frontmatter_keeps_the_catalogs_bytes():
    ours = CATALOG_V1.replace("keep\n", "keep\nmine\n")

    merged = merge_skill_markdown(CATALOG_V1, ours, CATALOG_V2)

    assert not merged.conflicted
    assert merged.text.startswith(CATALOG_V2.split("\n---\n")[0] + "\n---\n")
    assert merged.text.endswith("# Demo\n\nnew intro\nkeep\nmine\n")


def test_an_edit_next_to_the_catalogs_change_is_a_conflict_the_owner_wins():
    ours = CATALOG_V1.replace("intro\n", "intro\nmine\n")

    merged = merge_skill_markdown(CATALOG_V1, ours, CATALOG_V2)

    assert merged.conflicted
    assert "intro\nmine\nkeep" in merged.text
    assert "new intro" not in merged.text


def test_a_field_the_owner_added_survives_the_update():
    ours = CATALOG_V1.replace("version: 1\n", "version: 1\nallowed-tools: bash\n")

    merged = merge_skill_markdown(CATALOG_V1, ours, CATALOG_V2)

    front = yaml.safe_load(merged.text.split("\n---\n", 1)[0][4:])
    assert front["allowed-tools"] == "bash"
    assert front["description"] == "Second description"


def test_a_number_the_runtime_quoted_is_not_an_edit():
    ours = CATALOG_V1.replace("version: 1", "version: '1'")

    merged = merge_skill_markdown(CATALOG_V1, ours, CATALOG_V2)

    assert not merged.conflicted
    assert merged.text == CATALOG_V2


def test_text_without_frontmatter_merges_as_plain_text():
    merged = merge_skill_markdown("a\nb\n", "a\nb\nmine\n", "theirs\nb\n")

    assert merged.text == "theirs\nb\nmine\n"
    assert not merged.conflicted
