import pytest

from backend.copilot.tools.skills import (
    SKILL_ORIGIN_MARKETPLACE,
    SKILL_ORIGIN_USER,
    SkillWrite,
    _prepare_skill,
)

RAW_PACKAGE = """---
name: research
description: Research a topic
allowed-tools: WebSearch
metadata:
  author: Example Author
custom-key: preserve-this
---

# Research

Read references/method.md.
"""


def test_marketplace_package_preserves_exact_markdown_and_metadata():
    prepared = _prepare_skill(
        SkillWrite(
            name="research",
            description="Research a topic",
            body="# Research",
            skill_markdown=RAW_PACKAGE,
            files=[],
        ),
        SKILL_ORIGIN_MARKETPLACE,
    )
    assert prepared.rendered == RAW_PACKAGE
    assert prepared.parsed.extra["allowed-tools"] == "WebSearch"


def test_exact_package_preserves_crlf_bytes_and_remains_readable():
    raw = RAW_PACKAGE.replace("\n", "\r\n")
    prepared = _prepare_skill(
        SkillWrite("research", "Research", "Body", skill_markdown=raw, files=[]),
        SKILL_ORIGIN_MARKETPLACE,
    )
    assert prepared.rendered == raw
    assert prepared.parsed.name == "research"


def test_exact_markdown_must_match_destination_slug():
    with pytest.raises(ValueError, match="name"):
        _prepare_skill(
            SkillWrite("other-name", "Research", "Body", skill_markdown=RAW_PACKAGE),
            SKILL_ORIGIN_MARKETPLACE,
        )


def test_user_writes_cannot_bypass_normalized_storage():
    with pytest.raises(ValueError, match="marketplace"):
        _prepare_skill(
            SkillWrite("research", "Research", "Body", skill_markdown=RAW_PACKAGE),
            SKILL_ORIGIN_USER,
        )
