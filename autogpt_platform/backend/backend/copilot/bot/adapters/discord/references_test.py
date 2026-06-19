"""Tests for Discord reference extraction."""

from backend.copilot.bot.adapters.discord.references import (
    extract_referenced_channel_ids,
    replace_referenced_links,
)


def _extract(text: str, exclude: str = "0", limit: int = 3) -> list[str]:
    return extract_referenced_channel_ids(text, exclude_channel_id=exclude, limit=limit)


def test_extracts_channel_id_from_thread_link():
    text = "find my mentions in https://discord.com/channels/111/222/333 please"
    assert _extract(text) == ["222"]


def test_extracts_channel_mention():
    assert _extract("can you help with <#999>?") == ["999"]


def test_link_without_message_id():
    assert _extract("https://discord.com/channels/111/222") == ["222"]


def test_handles_subdomains_and_discordapp():
    text = (
        "old https://discordapp.com/channels/1/2/3 "
        "canary https://canary.discord.com/channels/1/4/5"
    )
    assert _extract(text) == ["2", "4"]


def test_dedupes_and_preserves_order():
    text = "<#10> https://discord.com/channels/1/20/3 <#10> <#30>"
    assert _extract(text) == ["10", "20", "30"]


def test_excludes_current_channel():
    text = "https://discord.com/channels/1/222/3 and <#222>"
    assert _extract(text, exclude="222") == []


def test_respects_limit():
    text = "<#1> <#2> <#3> <#4> <#5>"
    assert _extract(text, limit=2) == ["1", "2"]


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
