"""The slash-command frontmatter a skill carries into the index and the
chat's "/" picker."""

from backend.copilot.tools.skills import (
    _index_entry_from_metadata,
    invocation_frontmatter,
    parse_skill_markdown,
    render_skill_markdown,
)

_SKILL_MD = """---
name: fix-issue
description: Fix a GitHub issue by number
arguments: [issue]
argument-hint: [issue-number]
user-invocable: false
disable-model-invocation: true
---

Fix issue $issue.
"""


def test_slash_command_frontmatter_survives_parse_and_render():
    parsed = parse_skill_markdown(_SKILL_MD)
    assert parsed is not None
    assert parsed.extra["arguments"] == ["issue"]
    assert parsed.extra["argument-hint"] == ["issue-number"]
    assert parsed.extra["user-invocable"] is False
    assert parsed.extra["disable-model-invocation"] is True

    again = parse_skill_markdown(render_skill_markdown(parsed))
    assert again is not None
    assert dict(again.extra) == dict(parsed.extra)


def test_an_unquoted_hint_is_rendered_back_into_brackets():
    assert invocation_frontmatter({"argument-hint": ["issue-number"]}) == {
        "argument-hint": "[issue-number]"
    }
    assert invocation_frontmatter({"argument-hint": "[file] [format]"}) == {
        "argument-hint": "[file] [format]"
    }


def test_only_the_picker_fields_reach_the_index():
    assert invocation_frontmatter(
        {"user-invocable": False, "arguments": ["issue"], "license": "MIT"}
    ) == {"user-invocable": False}
    assert invocation_frontmatter({"user-invocable": True}) == {}


def test_an_index_entry_reads_the_picker_fields_back():
    entry = _index_entry_from_metadata(
        "fix-issue",
        {
            "kind": "copilot_skill",
            "description": "Fix a GitHub issue by number",
            "invocation": {"argument-hint": "[issue-number]", "user-invocable": False},
        },
    )
    assert entry is not None
    assert dict(entry.extra) == {
        "argument-hint": "[issue-number]",
        "user-invocable": False,
    }


def test_an_index_entry_from_before_the_picker_has_no_fields():
    entry = _index_entry_from_metadata(
        "fix-issue", {"kind": "copilot_skill", "description": "Fix an issue"}
    )
    assert entry is not None
    assert dict(entry.extra) == {}
