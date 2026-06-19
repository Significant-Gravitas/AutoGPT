"""Tests for Discord reference extraction."""

from backend.copilot.bot.adapters.discord.references import (
    ReferenceTarget,
    extract_referenced_targets,
    replace_referenced_links,
)


def _extract(
    text: str, exclude: str = "0", limit: int = 3
) -> list[tuple[str, str | None]]:
    targets = extract_referenced_targets(text, exclude_channel_id=exclude, limit=limit)
    return [(t.channel_id, t.message_id) for t in targets]


def test_extracts_channel_and_message_from_permalink():
    text = "find my mentions in https://discord.com/channels/111/222/333 please"
    assert _extract(text) == [("222", "333")]


def test_extracts_channel_mention_has_no_message():
    assert _extract("can you help with <#999>?") == [("999", None)]


def test_link_without_message_id():
    assert _extract("https://discord.com/channels/111/222") == [("222", None)]


def test_handles_subdomains_and_discordapp():
    text = (
        "old https://discordapp.com/channels/1/2/3 "
        "canary https://canary.discord.com/channels/1/4/5"
    )
    assert _extract(text) == [("2", "3"), ("4", "5")]


def test_dedupes_and_preserves_order():
    text = "<#10> https://discord.com/channels/1/20/3 <#10> <#30>"
    assert _extract(text) == [("10", None), ("20", "3"), ("30", None)]


def test_excludes_current_channel_for_bare_reference():
    text = "https://discord.com/channels/1/222 and <#222>"
    assert _extract(text, exclude="222") == []


def test_keeps_specific_message_even_in_current_channel():
    # "what was said here <permalink>" targets one message, so a permalink to
    # the current channel is a real request — not redundant context.
    text = "https://discord.com/channels/1/222/333"
    assert _extract(text, exclude="222") == [("222", "333")]


def test_respects_limit():
    text = "<#1> <#2> <#3> <#4> <#5>"
    assert _extract(text, limit=2) == [("1", None), ("2", None)]


def test_no_references_returns_empty():
    assert _extract("just a normal message, no links") == []


# ── replace_referenced_links ───────────────────────────────────────────


def test_replace_rewrites_link_to_readable_name():
    text = "read into https://discord.com/channels/111/222/333 please"
    out = replace_referenced_links(text, {"222": "Random Fact"})
    assert out == "read into #Random Fact please"


def test_replace_rewrites_channel_mention():
    out = replace_referenced_links("look at <#999> ok", {"999": "standup"})
    assert out == "look at #standup ok"


def test_replace_leaves_unknown_references_untouched():
    # Only channels we actually fetched (in the label map) get rewritten.
    text = "https://discord.com/channels/1/222/3 and <#777>"
    out = replace_referenced_links(text, {"222": "known"})
    assert "#known" in out
    assert "<#777>" in out


def test_replace_without_labels_is_noop():
    text = "https://discord.com/channels/1/2/3"
    assert replace_referenced_links(text, {}) == text


def test_reference_target_is_hashable():
    # Frozen model — equal values compare equal and collapse in a set.
    a = ReferenceTarget(channel_id="1", message_id="2")
    b = ReferenceTarget(channel_id="1", message_id="2")
    assert a == b
    assert len({a, b}) == 1
