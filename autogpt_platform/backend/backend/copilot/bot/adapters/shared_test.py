"""Tests for the platform-agnostic adapter helpers."""

import pytest

from .shared import InboundFile, collect_attachments, should_ignore


def _file(name="a.txt", size=10, mime="text/plain", content=b"data", fail=False):
    async def fetch() -> bytes:
        if fail:
            raise RuntimeError("download failed")
        return content

    return InboundFile(filename=name, size=size, mime_type=mime, fetch=fetch)


class TestCollectAttachments:
    @pytest.mark.asyncio
    async def test_keeps_files_under_caps(self):
        kept, skipped = await collect_attachments(
            [_file(content=b"hello")], max_count=10, max_bytes=1000
        )
        assert skipped == ()
        assert len(kept) == 1
        assert kept[0].filename == "a.txt"
        assert kept[0].content == b"hello"

    @pytest.mark.asyncio
    async def test_over_count_files_are_skipped(self):
        files = [_file(name=f"f{i}.txt") for i in range(12)]
        kept, skipped = await collect_attachments(files, max_count=10, max_bytes=1000)
        assert len(kept) == 10
        assert skipped == (
            ("f10.txt", "too many files attached"),
            ("f11.txt", "too many files attached"),
        )

    @pytest.mark.asyncio
    async def test_oversized_file_is_skipped(self):
        kept, skipped = await collect_attachments(
            [_file(size=2000)], max_count=10, max_bytes=1000
        )
        assert kept == ()
        assert skipped == (("a.txt", "too large"),)

    @pytest.mark.asyncio
    async def test_failed_download_is_skipped_not_fatal(self):
        kept, skipped = await collect_attachments(
            [_file(name="good.txt"), _file(name="bad.txt", fail=True)],
            max_count=10,
            max_bytes=1000,
        )
        assert [k.filename for k in kept] == ["good.txt"]
        assert skipped == (("bad.txt", "couldn't be downloaded"),)

    @pytest.mark.asyncio
    async def test_missing_name_and_mime_get_defaults(self):
        kept, _ = await collect_attachments(
            [_file(name=None, mime=None)], max_count=10, max_bytes=1000
        )
        assert kept[0].filename == "file"
        assert kept[0].mime_type == "application/octet-stream"


class TestShouldIgnore:
    def test_own_message_is_ignored(self):
        assert should_ignore(is_self=True, author_is_bot=True, bot_mentioned=True)

    def test_other_bot_without_mention_is_ignored(self):
        assert should_ignore(is_self=False, author_is_bot=True, bot_mentioned=False)

    def test_other_bot_with_mention_is_processed(self):
        assert not should_ignore(is_self=False, author_is_bot=True, bot_mentioned=True)

    def test_human_is_processed(self):
        assert not should_ignore(
            is_self=False, author_is_bot=False, bot_mentioned=False
        )
