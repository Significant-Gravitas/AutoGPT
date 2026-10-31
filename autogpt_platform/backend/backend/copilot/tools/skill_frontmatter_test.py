"""SKILL.md frontmatter is parsed within fixed limits."""

import pytest
import pytest_mock
import yaml

from backend.copilot.tools.skills import (
    ParsedSkill,
    parse_skill_markdown,
    render_skill_markdown,
)


def _skill_md(frontmatter: str) -> str:
    return f"---\n{frontmatter}\n---\n\n# Body\n"


def _nested_alias_frontmatter(levels: int = 5, width: int = 10) -> str:
    """Frontmatter whose name is built from nested aliases."""
    lines = [f"l0: &l0 [{', '.join(['lol'] * width)}]"]
    lines += [
        f"l{i}: &l{i} [{', '.join([f'*l{i - 1}'] * width)}]" for i in range(1, levels)
    ]
    lines += [f"name: *l{levels - 1}", "description: d"]
    return "\n".join(lines)


def test_nested_aliases_are_refused_before_anything_is_built(
    mocker: pytest_mock.MockerFixture,
) -> None:
    construct = mocker.spy(yaml.constructor.BaseConstructor, "construct_document")

    assert parse_skill_markdown(_skill_md(_nested_alias_frontmatter())) is None
    assert construct.call_count == 0


@pytest.mark.parametrize(
    "frontmatter",
    [
        "name: &n tidy\ndescription: d",
        "base: &b {x: 1}\nname: tidy\ndescription: d\nmetadata: *b",
    ],
    ids=["anchor", "alias"],
)
def test_anchors_and_aliases_are_refused(frontmatter: str) -> None:
    assert parse_skill_markdown(_skill_md(frontmatter)) is None


def test_ampersands_and_asterisks_inside_values_still_parse() -> None:
    parsed = parse_skill_markdown(
        _skill_md("name: tidy\ndescription: Tom & Jerry *work* together")
    )

    assert parsed is not None
    assert parsed.description == "Tom & Jerry *work* together"


def test_oversized_frontmatter_is_refused() -> None:
    frontmatter = f"name: tidy\ndescription: {'x' * (64 * 1024)}"

    assert parse_skill_markdown(_skill_md(frontmatter)) is None


def test_frontmatter_under_the_size_cap_still_parses() -> None:
    frontmatter = f"name: tidy\ndescription: d\nsource: {'x' * (60 * 1024)}"

    parsed = parse_skill_markdown(_skill_md(frontmatter))

    assert parsed is not None
    assert parsed.extra["source"] == "x" * (60 * 1024)


def test_frontmatter_over_the_expanded_size_cap_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backend.copilot.tools.skill_frontmatter.MAX_FRONTMATTER_EXPANDED_BYTES", 1024
    )
    triggers = "\n".join(f"  - trigger{i}" for i in range(200))

    assert (
        parse_skill_markdown(
            _skill_md(f"name: t\ndescription: d\ntriggers:\n{triggers}")
        )
        is None
    )


def test_deeply_nested_frontmatter_is_refused() -> None:
    frontmatter = f"name: tidy\ndescription: d\nmetadata: {'[' * 5000}{']' * 5000}"

    assert parse_skill_markdown(_skill_md(frontmatter)) is None


def test_ordinary_frontmatter_still_parses() -> None:
    parsed = parse_skill_markdown(
        _skill_md(
            "name: tidy\n"
            "description: Keep things tidy\n"
            "triggers:\n  - clean\n  - sort\n"
            "version: 2\n"
            "metadata:\n  owner: team\n  tags: [a, b]"
        )
    )

    assert parsed is not None
    assert parsed.name == "tidy"
    assert parsed.triggers == ("clean", "sort")
    assert parsed.version == "2"
    assert parsed.extra["metadata"] == {"owner": "team", "tags": ["a", "b"]}


def test_a_value_used_twice_renders_without_aliases_and_reads_back() -> None:
    shared = {"owner": "team"}
    skill = ParsedSkill(
        name="tidy",
        description="d",
        body="# Body",
        extra={"metadata": shared, "compatibility": shared},
    )

    rendered = render_skill_markdown(skill)
    parsed = parse_skill_markdown(rendered)

    assert "&" not in rendered and "*" not in rendered
    assert parsed is not None
    assert parsed.extra == {"metadata": shared, "compatibility": shared}
