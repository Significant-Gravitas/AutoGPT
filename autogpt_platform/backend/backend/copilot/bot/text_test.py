"""Tests for message batching + boundary splitting."""

from backend.data.sharing.workspace_refs import extract_artifact_links

from .text import _balance_code_fences, format_batch, split_at_boundary


class TestFormatBatch:
    def test_single_message_has_header(self):
        result = format_batch([("Bently", "123", "hello")], "discord")
        assert result == "[Message sent by Bently (Discord user ID: 123)]\nhello"

    def test_multi_message_labels_each_sender(self):
        result = format_batch(
            [
                ("Alice", "a1", "first"),
                ("Bob", "b2", "second"),
            ],
            "discord",
        )
        assert "[Multiple messages" in result
        assert "[From Alice (Discord user ID: a1)]\nfirst" in result
        assert "[From Bob (Discord user ID: b2)]\nsecond" in result

    def test_platform_name_is_capitalized(self):
        result = format_batch([("u", "1", "x")], "telegram")
        assert "Telegram user ID" in result


class TestSplitAtBoundary:
    def test_short_text_returns_unchanged(self):
        before, after = split_at_boundary("short", 100)
        assert before == "short"
        assert after == ""

    def test_splits_at_paragraph_boundary(self):
        text = "first paragraph.\n\nsecond paragraph that is long enough"
        before, after = split_at_boundary(text, 20)
        assert before == "first paragraph."
        assert after == "second paragraph that is long enough"

    def test_splits_at_newline_when_no_paragraph(self):
        text = "line one\nline two line three line four line five"
        before, after = split_at_boundary(text, 15)
        assert before == "line one"
        assert after == "line two line three line four line five"

    def test_splits_at_sentence_when_no_newline(self):
        text = "First sentence. Second sentence is quite a bit longer here."
        before, after = split_at_boundary(text, 20)
        assert before == "First sentence. "
        assert after == "Second sentence is quite a bit longer here."

    def test_falls_back_to_space_split(self):
        text = "word " * 50
        before, after = split_at_boundary(text, 30)
        assert not before.endswith(" ")
        # Rejoining drops one space at the cut, but no characters other
        # than whitespace should be lost.
        rejoined = (before + " " + after).replace("  ", " ").strip()
        assert rejoined == text.strip()

    def test_hard_cut_on_single_long_token(self):
        text = "a" * 500
        before, after = split_at_boundary(text, 100)
        assert len(before) == 100
        assert after == "a" * 400

    def test_does_not_split_inside_workspace_artifact_link(self):
        # A sentence boundary inside the display name would otherwise bisect
        # the `[name](workspace://id)` link and break artifact extraction.
        text = "x [report. final.txt](workspace://abc-123) y"
        before, after = split_at_boundary(text, 12)
        assert "workspace://" not in before
        assert "[report. final.txt](workspace://abc-123)" in after
        _, artifacts = extract_artifact_links(after)
        assert [a.file_id for a in artifacts] == ["abc-123"]

    def test_leading_long_artifact_link_does_not_stall(self):
        # A long link at the very start spanning the boundary must not return
        # an empty chunk (which would stall the stream) — emit the whole link.
        link = "[" + "report " * 30 + "final.txt](workspace://abc-123)"
        text = link + " and then more trailing text"
        before, after = split_at_boundary(text, 30)
        assert before != ""  # forward progress — never empty
        assert link in before  # whole link emitted intact
        assert "workspace://" not in after


class TestBalanceCodeFences:
    def test_balanced_code_unchanged(self):
        before = "prose\n```py\nprint('x')\n```\ntail"
        after = "more"
        b, a = _balance_code_fences(before, after)
        assert b == before
        assert a == after

    def test_open_fence_gets_closed_and_reopened(self):
        before = "prose\n```py\nprint('x')"
        after = "print('y')\n```\ntail"
        b, a = _balance_code_fences(before, after)
        assert b.endswith("```")
        assert a.startswith("```py\n")

    def test_reopens_with_no_lang_when_opener_had_none(self):
        before = "```\nsome code here"
        after = "more code\n```"
        b, a = _balance_code_fences(before, after)
        assert b.endswith("\n```")
        assert a.startswith("```\n")

    def test_preserves_latest_language_when_multiple_fences(self):
        before = "```py\nprint()\n```\nmiddle\n```ts\nconst x = 1"
        after = "const y = 2\n```"
        b, a = _balance_code_fences(before, after)
        assert b.endswith("```")
        assert a.startswith("```ts\n")


class TestSplitAtBoundaryWithCodeFences:
    def test_split_inside_fence_rebalances(self):
        code_block = "```python\n" + ("line\n" * 500) + "```\nafter"
        before, after = split_at_boundary(code_block, 300)
        # ``before`` must close the fence it opened.
        assert before.count("```") % 2 == 0
        # ``after`` must reopen with the same language tag.
        assert after.lstrip().startswith("```python")
